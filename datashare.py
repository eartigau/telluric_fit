"""
datashare.py — Share telluric-corrected data with collaborators on an HPC cluster.

OVERVIEW
--------
This script is the final step of the telluric correction pipeline.  Once the
pipeline has produced corrected spectra in tellupatched_<INSTRUMENT>/, this
script distributes them to individual collaborators:

  1. Reads the list of recipients and their assigned science targets from
     telluric_config.yaml  (data_recipients section).
  2. Locates each target's corrected spectra under
         {project_path}/tellupatched_{INSTRUMENT}/{target}_{batch}_smart/
     (falling back to {target}_{batch}/ or plain {target}/ if the _smart
     variant does not exist).
  3. Copies new/changed files into a per-user directory:
         {datashare_dir}/{username}/{target}/
     A FITS CHECKSUM comparison (with mtime-backed cache) avoids redundant
     copies of large files.
  4. Prunes target subdirs that are no longer assigned to the user (e.g. when
     targets are reassigned between batches) so stale data does not accumulate.
  5. Sets POSIX ACLs via setfacl so each user can read only their own folder:
         setfacl -m u:{user}:r-x  {datashare_dir}          # traverse the root
         setfacl -m u:{user}:r-x  {datashare_dir}/..       # traverse parent dirs
         setfacl -R -m u:{user}:r-x {datashare_dir}/{user} # read their subtree
     NOTE ON PERMISSIONS: users also need execute permission on every directory
     *above* datashare_dir in the path.  On /scratch this means all parent dirs
     up to / must be traversable.  See set_acl() for the full parent-dir walk.
  6. Writes a plain-text email per user summarising available targets and the
     rsync command needed to download them.
  7. Optionally sends those emails via Gmail SMTP (requires a Gmail App Password
     stored in the macOS Keychain or ~/.gmail_app_password).

RECIPIENT FORMAT in telluric_config.yaml
-----------------------------------------
  data_recipients:
    username:                    # HPC login name used for ACL and directory name
      name: Full Name            # used in email greeting
      email: addr@host.tld       # used to send/write the notification email
      targets:                   # list of science targets assigned to this user
        - TARGET1
        - TARGET2

  Legacy form (list instead of dict) is also accepted:
    username:
      - TARGET1
      - TARGET2

CHECKSUM CACHE
--------------
  {datashare_dir}/.checksum_cache.csv stores (path, mtime, FITS CHECKSUM)
  rows.  Before reading a FITS header the cache is checked: if the file's
  mtime has not changed the cached checksum is returned immediately, avoiding
  slow header reads for thousands of files.

Usage
-----
    python datashare.py                     # copy + prune stale + set ACL + write emails
    python datashare.py --dry-run           # preview without copying or removing
    python datashare.py --user alexsm       # only process one recipient
    python datashare.py --email-only        # only write/print email summaries
    python datashare.py --send-email        # interactive menu to send emails via Gmail
    python datashare.py --instrument SPIROU # override instrument
    python datashare.py --no-prompt         # skip interactive email-address prompts
    python datashare.py --list-recipients   # list all recipients with name, email and targets
"""

import argparse
import csv
import json
import os
from datetime import datetime
import shutil
import smtplib
import subprocess
import sys
from email.mime.text import MIMEText
import yaml
from tqdm import tqdm
from astropy.io import fits

# Absolute path of this script's directory — used as a fallback project root
# and to locate telluric_config.yaml when no machine entry matches.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the master YAML config shared by all scripts in this pipeline.
CFG_PATH = os.path.join(SCRIPT_DIR, 'telluric_config.yaml')

# Gmail address used as the From: address when sending notification emails.
# Must match the account whose App Password is stored in the keychain.
SENDER_EMAIL = 'etienne.artigau@gmail.com'


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def get_science_targets(config):
    """Return the sorted unique list of science targets from all data_recipients.

    Walks all recipient entries in the data_recipients section of the YAML and
    collects every target name into a single de-duplicated set.  This replaces
    the legacy top-level 'science_targets' key that used to be maintained by
    hand — now the master list is derived automatically so it is always in sync
    with recipient assignments.
    """
    raw = config.get('data_recipients', {})
    targets = set()
    for value in raw.values():
        if isinstance(value, list):
            # Legacy form: the value is directly a list of target names
            targets.update(value)
        elif isinstance(value, dict):
            # New form: the value is a dict with 'targets', 'name', 'email' keys
            targets.update(value.get('targets', []))
    return sorted(targets)


def load_config():
    """Load and return the YAML config as a plain Python dict.

    All other functions receive the config dict rather than re-reading the
    file, so the file is only opened once per run.
    """
    with open(CFG_PATH) as fh:
        return yaml.safe_load(fh)


def save_config(config):
    """Write the config dict back to telluric_config.yaml.

    WARNING: PyYAML does not preserve comments.  Only call this function when
    the user has explicitly provided new information (name / email) that needs
    to be persisted.  The --no-prompt flag suppresses all interactive prompts
    and therefore also suppresses writes.
    """
    with open(CFG_PATH, 'w') as fh:
        yaml.dump(config, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)


def parse_recipients(raw_recipients):
    """Normalise data_recipients to a uniform dict structure.

    Returns:
        {username: {'name': str|None, 'email': str|None, 'targets': list}}

    The YAML supports two forms for each recipient:

      Legacy list form (name/email unknown):
        alexsm:
          - PROXIMA
          - GL699

      New dict form (all fields present):
        alexsm:
          name: Alejandro Suárez Mascareño
          email: asm@iac.es
          targets:
            - PROXIMA
            - GL699

    Both are normalised to the same output structure so the rest of the code
    never has to check which form was used.
    """
    result = {}
    for user, value in raw_recipients.items():
        if isinstance(value, list):
            # Legacy form: promote to dict with unknown name/email
            result[user] = {'name': None, 'email': None, 'targets': value}
        else:
            result[user] = {
                'name': value.get('name') or None,
                'email': value.get('email') or None,
                'targets': value.get('targets', []),
            }
    return result


def prompt_missing_info(parsed, config, no_prompt=False):
    """Interactively ask for any missing name or email address and persist to YAML.

    Called once at startup after parse_recipients().  For each recipient that
    is missing a name or email the user is prompted on stdin.  Answers are
    written back to telluric_config.yaml via save_config() so that future
    runs do not need to prompt again.

    If no_prompt=True (--no-prompt flag) a warning is printed instead and the
    missing fields remain empty.  This is safe for automated/batch runs where
    stdin is not a terminal.
    """
    cfg_recipients = config.setdefault('data_recipients', {})
    changed = False
    for user, info in parsed.items():
        # If the raw entry was still in legacy list form, promote it so we can
        # add name/email keys without losing the target list.
        entry = cfg_recipients.get(user)
        if isinstance(entry, list):
            entry = {'targets': entry}
            cfg_recipients[user] = entry

        if not info['name']:
            if no_prompt:
                print(f'  [WARN] No name for {user} — pass --no-prompt to silence')
            else:
                name = input(f'  Full name for {user}: ').strip()
                if name:
                    info['name'] = name
                    entry['name'] = name
                    changed = True

        if not info['email']:
            if no_prompt:
                print(f'  [WARN] No email for {user} — pass --no-prompt to silence')
            else:
                addr = input(f'  Email address for {user}: ').strip()
                if addr:
                    info['email'] = addr
                    entry['email'] = addr
                    changed = True

    if changed:
        save_config(config)
        print('  [INFO] Contact info saved to telluric_config.yaml')


def get_gmail_app_password():
    """Retrieve the Gmail App Password from macOS Keychain or ~/.gmail_app_password.

    Two sources are tried in order:
      1. macOS Keychain — stored with:
             security add-generic-password -a etienne.artigau@gmail.com \
                 -s gmail_app_password -w <password>
         This is the preferred method: the password never sits in a plain file.
      2. Plain text file ~/.gmail_app_password — a single line containing the
         App Password.  Less secure but works on Linux HPC nodes.

    Returns the password string, or None if neither source is available.
    The caller (send_gmail) will print an error and return False when None.
    """
    try:
        result = subprocess.run(
            ['security', 'find-generic-password',
             '-a', SENDER_EMAIL, '-s', 'gmail_app_password', '-w'],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            pw = result.stdout.strip()
            if pw:
                return pw
    except FileNotFoundError:
        # 'security' binary not available (Linux HPC), fall through to file
        pass
    pw_file = os.path.expanduser('~/.gmail_app_password')
    if os.path.exists(pw_file):
        with open(pw_file) as fh:
            return fh.read().strip()
    return None


def send_gmail(recipient_email, subject, body):
    """Send a single plain-text email via Gmail SMTP over TLS (port 465).

    Uses the App Password retrieved by get_gmail_app_password().  A Gmail
    App Password is a 16-character code generated at
    https://myaccount.google.com/apppasswords — it allows SMTP access even
    when 2-Step Verification is enabled, without exposing the main password.

    Returns True on success, False on any failure (missing password, network
    error, authentication failure, etc.).  Errors are printed to stdout so
    the caller can keep processing other recipients.
    """
    app_password = get_gmail_app_password()
    if not app_password:
        print('  [ERROR] Gmail App Password not found — cannot send email.')
        return False
    msg = MIMEText(body, 'plain', 'utf-8')
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = recipient_email
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, app_password)
            smtp.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
        return True
    except Exception as exc:
        print(f'  [ERROR] Failed to send to {recipient_email}: {exc}')
        return False


# Path to the JSON file that records when each recipient was last emailed.
# Stored next to this script so it persists across runs.
EMAIL_LOG_PATH = os.path.join(SCRIPT_DIR, 'email_log.json')


def _load_email_log():
    """Load the email log JSON from disk.  Returns an empty dict on any error.

    The log maps {username: 'YYYY-MM-DD HH:MM UTC'} and is displayed in the
    interactive send menu so the operator knows who was already notified and when.
    """
    if os.path.exists(EMAIL_LOG_PATH):
        try:
            with open(EMAIL_LOG_PATH) as fh:
                return json.load(fh)
        except Exception:
            pass
    return {}


def _save_email_log(log):
    """Persist the email log dict back to disk as JSON."""
    with open(EMAIL_LOG_PATH, 'w') as fh:
        json.dump(log, fh, indent=2)


def interactive_send_emails(email_blocks):
    """Show a numbered menu of recipients and let the operator choose whom to email.

    email_blocks is a list of (user, name, addr, subject, body) tuples built
    earlier in main().  The function:
      1. Prints a numbered table with each recipient's display name, email
         address, and the timestamp of the last email sent (from email_log.json).
      2. Asks the operator to enter space-separated numbers, "all", or "none".
      3. Calls send_gmail() for each selected recipient.
      4. Updates email_log.json with the current UTC timestamp for every
         successful send so future runs show the correct "last sent" note.
    """
    if not email_blocks:
        print('No emails to send.')
        return

    email_log = _load_email_log()

    print()
    print('=' * 60)
    print('SELECT RECIPIENTS TO EMAIL')
    print('=' * 60)
    for i, (user, name, addr, _subj, _body) in enumerate(email_blocks, 1):
        display = name if name else user
        addr_str = addr if addr else '(no email address)'
        last_sent = email_log.get(user)
        note = '  [last sent: {}]'.format(last_sent) if last_sent else ''
        print('  {:2d}.  {:<30}  {}{}'.format(i, display, addr_str, note))
    print()
    print('Enter numbers separated by spaces (e.g. "1 3 5"), "all", or "none":')
    raw = input('> ').strip().lower()

    if raw == 'none' or raw == '':
        print('No emails sent.')
        return

    if raw == 'all':
        chosen = list(range(len(email_blocks)))
    else:
        chosen = []
        for tok in raw.split():
            try:
                idx = int(tok) - 1
                if 0 <= idx < len(email_blocks):
                    chosen.append(idx)
                else:
                    print(f'  [WARN] Ignoring out-of-range number: {tok}')
            except ValueError:
                print(f'  [WARN] Ignoring invalid input: {tok}')

    print()
    for idx in chosen:
        user, name, addr, subj, body = email_blocks[idx]
        display = name if name else user
        if not addr:
            print('  [{}] SKIP — no email address'.format(display))
            continue
        print('  Sending to {} <{}> … '.format(display, addr), end='', flush=True)
        ok = send_gmail(addr, subj, body)
        print('OK' if ok else 'FAILED')
        if ok:
            email_log[user] = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    _save_email_log(email_log)


def get_project_path(config):
    """Return the project_path for the current machine by probing detect_path entries.

    The machines section of the YAML maps logical machine names to their
    settings.  Each entry has a 'detect_path' — a file or directory that
    exists only on that machine (e.g. /home/eartigau/.fir_token on the FIR
    cluster, /Users/eartigau/telluric_fit on the MacBook).

    The first machine whose detect_path exists on disk wins.  This lets the
    same config and script run unchanged on multiple machines with different
    filesystem layouts.

    Falls back to SCRIPT_DIR if no machine matches (e.g. a new machine not yet
    in the config).
    """
    machines = config.get('machines', {})
    for _name, mcfg in machines.items():
        detect = mcfg.get('detect_path', '')
        if detect and os.path.exists(detect):
            return mcfg['project_path']
    return SCRIPT_DIR


def get_hostname(config):
    """Return the rsync-accessible hostname of the current machine, or None.

    Used when building the rsync command in notification emails so recipients
    get the correct server address without having to know the hostname themselves.
    If the current machine has no 'hostname' entry in the config (e.g. the
    MacBook), returns None and the email template shows a placeholder instead.
    """
    machines = config.get('machines', {})
    for _name, mcfg in machines.items():
        detect = mcfg.get('detect_path', '')
        if detect and os.path.exists(detect):
            return mcfg.get('hostname', None)
    return None


# ---------------------------------------------------------------------------
# Checksum cache
# ---------------------------------------------------------------------------

class ChecksumCache:
    """CSV-backed cache mapping file path → (mtime, FITS CHECKSUM keyword value).

    WHY THIS EXISTS
    ---------------
    Each user can have hundreds of FITS files.  Reading the CHECKSUM keyword
    from each header before deciding whether to copy is slow (astropy opens the
    file, reads the HDU structure, etc.).  If the file's mtime has not changed
    since the last run the content cannot have changed either, so we can reuse
    the cached CHECKSUM without opening the file again.

    CACHE FILE FORMAT
    -----------------
    A CSV at {datashare_dir}/.checksum_cache.csv with columns:
        path      — absolute path to the file
        mtime     — modification time at last read (float, seconds since epoch)
        checksum  — value of the FITS CHECKSUM keyword (may be empty string)

    LIFECYCLE
    ---------
    - __init__  : loads the CSV from disk (creates empty cache if missing)
    - get(path) : returns cached checksum if mtime still matches, else None
    - set(path) : stores checksum + current mtime
    - save()    : rewrites the CSV (no-op if nothing changed since last load)
    """

    def __init__(self, cache_path):
        self.path = cache_path
        self._cache = {}   # str -> (float mtime, str|None checksum)
        self._dirty = False
        self._load()

    def _load(self):
        """Parse the CSV file into self._cache.  Silently ignores corrupt rows."""
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, newline='') as fh:
                for row in csv.DictReader(fh):
                    try:
                        self._cache[row['path']] = (
                            float(row['mtime']),
                            row['checksum'] or None,
                        )
                    except (KeyError, ValueError):
                        pass
        except OSError:
            pass

    def get(self, filepath):
        """Return cached checksum if mtime still matches, else None.

        A return value of None means the caller must read the FITS header and
        then call set() to populate the cache for next time.
        """
        entry = self._cache.get(filepath)
        if entry is None:
            return None
        cached_mtime, cached_cs = entry
        try:
            current_mtime = os.path.getmtime(filepath)
        except OSError:
            return None
        if abs(current_mtime - cached_mtime) < 1e-3:
            return cached_cs
        return None  # file was modified since last read — treat as cache miss

    def set(self, filepath, checksum):
        """Store checksum together with the file's current mtime."""
        try:
            mtime = os.path.getmtime(filepath)
        except OSError:
            return
        self._cache[filepath] = (mtime, checksum)
        self._dirty = True

    def save(self):
        """Persist cache to disk (no-op if nothing changed since last load/save)."""
        if not self._dirty:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=['path', 'mtime', 'checksum'])
            writer.writeheader()
            for fp, (mtime, cs) in sorted(self._cache.items()):
                writer.writerow({'path': fp, 'mtime': f'{mtime:.6f}', 'checksum': cs or ''})
        self._dirty = False


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def find_target_dir(tellupatched_dir, target, batch_name):
    """Resolve the on-disk source folder for a target, trying three name variants.

    The pipeline can produce output directories under three naming conventions:
      1. {target}_{batch_name}_smart  — smart template run (preferred)
      2. {target}_{batch_name}        — model-only run
      3. {target}                     — legacy naming (no batch suffix)

    Returns the path of the first variant that exists as a directory, or None
    if none of them exist (target not yet processed or processed elsewhere).
    """
    for candidate in [
        f'{target}_{batch_name}_smart',
        f'{target}_{batch_name}',
        target,
    ]:
        path = os.path.join(tellupatched_dir, candidate)
        if os.path.isdir(path):
            return path
    return None


def collect_target_files(tellupatched_dir, target, batch_name):
    """Return sorted list of all files to share for a given target.

    Walks the target's source directory recursively and returns absolute paths
    to every file found.  Returns an empty list if the target directory does
    not exist (i.e. the target has not been processed yet).
    """
    target_dir = find_target_dir(tellupatched_dir, target, batch_name)
    if target_dir is None:
        return []
    files = []
    for root, _dirs, fnames in os.walk(target_dir):
        for fname in fnames:
            files.append(os.path.join(root, fname))
    return sorted(files)


def fits_checksum(path, cache=None):
    """Return the CHECKSUM value from the primary HDU FITS header, or None.

    The FITS CHECKSUM keyword (FITS Standard section 4.4.2.7) is a
    base-64–encoded value that summarises the entire HDU.  It is written by
    astropy when saving a file and changes whenever the data or headers change,
    making it a reliable change-detection key that is faster to compare than a
    full MD5 of the file.

    If *cache* is provided:
      - Check cache first; if the mtime matches, return cached value immediately.
      - After a fresh header read, store the result in the cache for next time.
    """
    if cache is not None:
        cached = cache.get(path)
        if cached is not None:
            return cached
    try:
        hdr = fits.getheader(path, ext=0)
        cs = hdr.get('CHECKSUM', None)
    except Exception:
        cs = None
    if cache is not None:
        cache.set(path, cs)
    return cs


def needs_copy(src, dst, cache=None):
    """Return True if src should be copied to dst.

    Decision logic:
      - dst does not exist yet                       → copy (obvious)
      - src or dst lacks a FITS CHECKSUM keyword     → copy (can't compare reliably)
      - CHECKSUM values differ                       → copy (content changed)
      - CHECKSUM values match                        → skip (identical file already there)

    Using FITS CHECKSUM rather than file size or mtime avoids both false
    positives (same content, different mtime after re-extraction) and false
    negatives (same mtime, different content due to pipeline re-run).
    """
    if not os.path.exists(dst):
        return True
    src_cs = fits_checksum(src, cache)
    dst_cs = fits_checksum(dst, cache)
    if src_cs is None or dst_cs is None:
        return True
    return src_cs != dst_cs


def copy_target(src_dir, target, dest_user_dir, dry_run, batch_name, cache=None):
    """Copy all files for one target into dest_user_dir/{target}/.

    SOURCE LAYOUT
    -------------
    src_dir/
      {target}_{batch_name}_smart/      ← preferred variant
        file1.fits
        subdir/
          file2.fits

    DESTINATION LAYOUT
    ------------------
    dest_user_dir/
      {target}/                         ← always flat target name (no batch suffix)
        file1.fits
        subdir/
          file2.fits

    The destination directory is created if it does not exist.  A tqdm
    progress bar shows copy/skip counts per target.  In dry_run mode the
    target dir is not created and no files are copied; only the file count is
    returned.

    Returns (n_files, found):
      n_files — number of files in the source directory (0 if not found)
      found   — True if the source directory exists, False otherwise
    """
    src_target = find_target_dir(src_dir, target, batch_name)
    dst_target = os.path.join(dest_user_dir, target)
    if src_target is None:
        return 0, False   # target not processed yet

    files = collect_target_files(src_dir, target, batch_name)
    n_total = len(files)
    if not dry_run:
        os.makedirs(dst_target, exist_ok=True)
        n_copied = 0
        n_skipped = 0
        with tqdm(files, desc=target, unit='file', leave=False) as pbar:
            for fpath in pbar:
                # Preserve relative sub-path structure within the target dir
                rel = os.path.relpath(fpath, src_target)
                dst = os.path.join(dst_target, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if needs_copy(fpath, dst, cache):
                    shutil.copy2(fpath, dst)
                    # Warm the cache for the destination file immediately so
                    # the next run can skip it without re-reading its header.
                    if cache is not None:
                        fits_checksum(dst, cache)
                    n_copied += 1
                else:
                    n_skipped += 1
                pbar.set_postfix(copied=n_copied, skip=n_skipped)
    return n_total, True


def prune_user_dir(user_dir, expected_targets, dry_run):
    """Remove subdirectories in user_dir that are no longer in expected_targets.

    This cleans up stale target data when targets are reassigned between
    pipeline runs (e.g. a target moves from one collaborator to another, or is
    dropped entirely from the batch).

    Returns a list of directory names that were removed (or would be in
    dry_run mode).  Only immediate subdirectories are considered — files
    directly in user_dir are left untouched.
    """
    removed = []
    if not os.path.isdir(user_dir):
        return removed
    for entry in sorted(os.listdir(user_dir)):
        full = os.path.join(user_dir, entry)
        if not os.path.isdir(full):
            continue
        if entry not in expected_targets:
            removed.append(entry)
            if not dry_run:
                shutil.rmtree(full)
    return removed


def set_acl(basedir, user_dir, user, dry_run):
    """Set POSIX ACLs so *user* can traverse basedir and read their subtree.

    WHY setfacl IS NEEDED
    ---------------------
    On Compute Canada clusters (and most HPC systems) each user owns their own
    home/scratch/project directories and other users cannot read them by default.
    Standard Unix permissions (chmod) can only express one owner, one group, and
    "other".  POSIX ACLs (setfacl/getfacl) allow per-user permission entries on
    top of the standard permission bits without changing ownership.

    WHAT THIS FUNCTION DOES
    -----------------------
    1. Walk every parent directory from basedir up to the filesystem root and
       grant the user r-x (read + traverse/execute) on each one.  Without this,
       the user's shell cannot even reach basedir, regardless of what permissions
       their own subdirectory has.  This is the most common source of "Permission
       denied" errors on /scratch paths on HPC clusters.

    2. Grant r-x on basedir itself (the shared data root, e.g. /scratch/eartigau/
       telluric_share/).

    3. Recursively grant r-x on the user's personal subdirectory
       (basedir/{user}/) and everything inside it.

    COMMANDS RUN (per user)
    -----------------------
      For each ancestor dir above basedir:
        setfacl -m u:{user}:r-x /path/to/ancestor

      For basedir and user_dir:
        setfacl -m u:{user}:r-x {basedir}
        setfacl -R -m u:{user}:r-x {user_dir}

    NOTE ON ACL MASK
    ----------------
    On Linux the "effective permission mask" entry (mask::) can silently cap the
    effective permissions even when a user entry says r-x.  If another setfacl
    call later narrows the mask, previously granted permissions become ineffective.
    To avoid this, always run setfacl with the exact permissions needed and check
    `getfacl {user_dir}` if users still report access issues.

    SCRATCH-SPECIFIC CAVEAT
    -----------------------
    On Alliance/CC Lustre /scratch filesystems the sticky bit and quotas mean the
    top-level /scratch/{owner}/ directory is owned by the owner and its permissions
    are not normally writable by root-owned setfacl.  If setfacl fails on an
    ancestor, a warning is printed and the function continues — the operator must
    manually ensure the top-level directories are traversable (e.g. chmod o+x
    /scratch/eartigau/).
    """
    if dry_run:
        return

    def _run_setfacl(cmd):
        """Run a setfacl command, printing a warning on failure."""
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError:
            print(f'  [WARN] setfacl not found — skipping ACL for {user}')
            return False
        except subprocess.CalledProcessError as exc:
            print(f'  [WARN] setfacl failed: {" ".join(cmd)}: {exc.stderr.strip()}')
            return False
        return True

    # Step 1: Grant traverse (r-x) on every ancestor directory above basedir.
    # Without this the user's shell returns "Permission denied" before it even
    # reaches the data, regardless of what is set on basedir or user_dir.
    basedir_abs = os.path.abspath(basedir)
    ancestor = os.path.dirname(basedir_abs)
    while ancestor != os.path.dirname(ancestor):  # stop at filesystem root
        if not _run_setfacl(['setfacl', '-m', f'u:{user}:r-x', ancestor]):
            break  # if we can't set ACL on an ancestor, don't bother going higher
        ancestor = os.path.dirname(ancestor)

    # Step 2: Grant r-x on the shared data root (basedir) so the user can list
    # or cd into it and see their own subdirectory entry.
    if not _run_setfacl(['setfacl', '-m', f'u:{user}:r-x', basedir_abs]):
        return

    # Step 3: Recursively grant r-x on the user's personal subdirectory.
    # -R applies the ACL to all existing files and directories underneath.
    # New files created later will NOT automatically inherit this ACL unless
    # default ACLs are also set (setfacl -d), but we don't need that here
    # because this function is re-run after every pipeline execution.
    _run_setfacl(['setfacl', '-R', '-m', f'u:{user}:r-x', user_dir])


def build_email(user, name, targets, batch_name, basedir, instrument, missing, hostname=None):
    """Return a (subject, body) tuple for a recipient's notification email.

    The email tells the recipient:
      - Where their data lives on the server (absolute path).
      - The exact rsync command to download it to their laptop.
      - How many targets were found and their names.
      - Which assigned targets had no data on disk yet (so they know what is
        still pending and won't think we forgot them).

    Parameters
    ----------
    user        : HPC username (used as their subdirectory name under basedir)
    name        : Full name (used in greeting) or None
    targets     : List of target names that were actually copied
    batch_name  : Batch identifier string from the YAML (e.g. 'v28Apr2026')
    basedir     : Absolute path to the shared data root on the server
    instrument  : 'NIRPS' or 'SPIROU'
    missing     : List of targets assigned to the user but not found on disk
    hostname    : SSH hostname for the rsync command (None → placeholder text)
    """
    user_path = os.path.join(basedir, user)
    if hostname:
        rsync_cmd = f'rsync -avz {user}@{hostname}:{user_path}/ ./data_from_fir/'
    else:
        # Hostname not configured for this machine — show a placeholder so the
        # operator can fill it in before forwarding the email.
        rsync_cmd = f'rsync -avz <server>:{user_path}/ ./data_from_fir/'

    # Use first name for a more personal greeting; fall back to username
    first_name = name.split()[0] if name else user
    subject = f'{instrument} telluric-corrected data available (batch {batch_name})'
    lines = [
        f'Dear {first_name},',
        '',
        f'Your telluric-corrected {instrument} data for batch {batch_name} is now available at:',
        f'  {user_path}/',
        '',
        'To download your data, run the following command on your local machine:',
        f'  {rsync_cmd}',
        '',
    ]

    if targets:
        lines.append('{} target(s): {}'.format(len(targets), ', '.join(targets)))
    else:
        lines.append('No targets were found on disk for you in this batch.')

    if missing:
        lines.append('')
        lines.append('Note: the following targets were attributed to you but had no data on disk:')
        for t in missing:
            lines.append(f'  - {t}')

    lines += [
        '',
        'Best regards,',
        'Étienne',
    ]
    return subject, '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _recipients_epilog():
    """Build a formatted recipients table for the argparse --help epilog.

    Reads the current config and formats a table of all recipients with their
    username, full name, email, and targets.  Also appends example invocations
    using the first recipient's username so the operator can copy-paste them.
    Returns an empty string if the config cannot be read (e.g. running --help
    in a directory without a telluric_config.yaml).
    """
    try:
        config = load_config()
        parsed = parse_recipients(config.get('data_recipients', {}))
    except Exception:
        return ''
    header = '  {:<12}  {:<25}  {:<40}  Targets'.format('User', 'Name', 'Email')
    lines = ['', 'Recipients:', header, '  ' + '-' * 106]
    first_user = None
    for user, info in sorted(parsed.items()):
        if first_user is None:
            first_user = user
        name    = info.get('name')  or '(missing)'
        email   = info.get('email') or '(missing)'
        targets = ', '.join(info.get('targets', []))
        lines.append('  {:<12}  {:<25}  {:<40}  {}'.format(user, name, email, targets))
    if first_user:
        lines += [
            '',
            'Examples:',
            '  python datashare.py                        # copy data for all recipients',
            '  python datashare.py --user {}    # copy data for one recipient'.format(first_user),
            '  python datashare.py --user {} --send-email # copy + send email'.format(first_user),
            '  python datashare.py --send-email           # interactive menu to send emails',
            '  python datashare.py --dry-run              # preview without copying',
        ]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Share telluric-corrected data with collaborators.',
        epilog=_recipients_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--instrument', default=None,
                        help='Override instrument (NIRPS or SPIROU)')
    parser.add_argument('--user', default=None,
                        help='Process only this recipient')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be copied without doing it')
    parser.add_argument('--email-only', action='store_true',
                        help='Only print email summaries, no copy/ACL')
    parser.add_argument('--send-email', action='store_true',
                        help='Interactive menu to send emails via Gmail SMTP')
    parser.add_argument('--no-prompt', action='store_true',
                        help='Skip interactive prompts for missing name/email')
    parser.add_argument('--list-recipients', action='store_true',
                        help='List all recipients with name, email and targets, then exit')
    args = parser.parse_args()

    config = load_config()

    # --list-recipients: quick tabular summary and exit, no copying or ACL work
    if args.list_recipients:
        raw = config.get('data_recipients', {})
        parsed = parse_recipients(raw)
        print('{:<12}  {:<25}  {:<40}  Targets'.format('User', 'Name', 'Email'))
        print('-' * 110)
        for user, info in sorted(parsed.items()):
            name  = info.get('name')  or '(missing)'
            email = info.get('email') or '(missing)'
            targets = ', '.join(info.get('targets', []))
            print('{:<12}  {:<25}  {:<40}  {}'.format(user, name, email, targets))
        sys.exit(0)

    # Resolve instrument: CLI flag overrides YAML, default is NIRPS.
    # project_path is the root of the telluric pipeline output on this machine
    # (resolved via the machines section; falls back to SCRIPT_DIR).
    instrument = (args.instrument or config.get('instrument', 'NIRPS')).upper()
    project_path = get_project_path(config)
    hostname = get_hostname(config)
    batch_name = config.get('batch', {}).get('name', 'unknown')

    raw_recipients = config.get('data_recipients', {})
    if not raw_recipients:
        print('No data_recipients defined in telluric_config.yaml. Nothing to do.')
        sys.exit(0)

    # --user: restrict processing to a single recipient (useful for re-sending
    # one email or re-copying data for one person without touching others)
    if args.user:
        if args.user not in raw_recipients:
            print(f'Error: user "{args.user}" not found in data_recipients.')
            sys.exit(1)
        raw_recipients = {args.user: raw_recipients[args.user]}

    recipients = parse_recipients(raw_recipients)
    # Prompt for missing name/email before starting the (potentially slow) copy
    # loop so the operator doesn't wait for everything to finish only to be asked
    # for info at the end.
    prompt_missing_info(recipients, config, no_prompt=args.no_prompt)

    # Source directory: contains one subdirectory per target, named
    #   {target}_{batch_name}_smart  or  {target}_{batch_name}  or  {target}
    tellupatched_dir = os.path.join(project_path, f'tellupatched_{instrument}')

    # Destination root: each recipient gets a subdirectory named after their
    # HPC username.  Defaults to ./data_dir/ next to the script if datashare_dir
    # is not set in the YAML.  On FIR this should be set to a path under
    # /scratch or /project that is accessible by the cluster login nodes via SSH.
    basedir = config.get('datashare_dir', os.path.join(SCRIPT_DIR, 'data_dir'))

    if not args.dry_run and not args.email_only:
        os.makedirs(basedir, exist_ok=True)

    if not args.email_only:
        print(f'Instrument   : {instrument}')
        print(f'Batch        : {batch_name}')
        print(f'Source       : {tellupatched_dir}')
        print(f'Destination  : {basedir}')
        if args.dry_run:
            print('Mode         : DRY RUN (nothing will be copied)')
        print()

    # Checksum cache: shared across all users and targets in this run to avoid
    # re-reading FITS headers for files that haven't changed.
    cache_path = os.path.join(basedir, '.checksum_cache.csv')
    cache = ChecksumCache(cache_path)

    # email_dir: plain-text email files are saved here (one per user per batch)
    # so the operator can review or forward them manually if --send-email is not used.
    email_dir = os.path.join(basedir, 'emails')
    if not args.dry_run and not args.email_only:
        os.makedirs(email_dir, exist_ok=True)

    # email_blocks accumulates (user, name, addr, subject, body) for every
    # recipient so we can print the email summary and optionally send emails
    # after all copying is done.
    email_blocks = []
    # acl_pending is kept for potential future use (e.g. deferred ACL application).
    # Currently ACLs are applied immediately after each user's copy completes.
    acl_pending = []

    # -----------------------------------------------------------------------
    # Per-recipient copy + ACL loop
    # -----------------------------------------------------------------------
    for user, info in sorted(recipients.items()):
        # Each user gets a private subdirectory: {basedir}/{username}/
        user_dir = os.path.join(basedir, user)
        targets = info['targets']
        name = info['name']
        found_targets = []    # targets that exist on disk and were copied
        missing_targets = []  # targets assigned to user but not yet on disk
        total_files = 0

        if not args.email_only:
            label = f'{name} ({user})' if name else user
            print(f'[{label}]')

        # Copy each assigned target into the user's directory
        for target in targets:
            n_files, found = copy_target(
                tellupatched_dir, target, user_dir,
                args.dry_run or args.email_only, batch_name, cache,
            )
            if found:
                found_targets.append(target)
                total_files += n_files
                if not args.email_only:
                    tag = '(dry)' if args.dry_run else 'copied'
                    print(f'  {tag}  {target}  ({n_files} file{"s" if n_files != 1 else ""})')
            else:
                missing_targets.append(target)
                if not args.email_only:
                    print(f'  [MISS]  {target}  (not found in {tellupatched_dir})')

        # Remove stale target directories no longer assigned to this user.
        # This handles the case where a target moves to a different recipient
        # between pipeline runs — without pruning, the old data would accumulate.
        if not args.email_only:
            pruned = prune_user_dir(user_dir, set(targets), args.dry_run)
            for stale in pruned:
                tag = '(dry)' if args.dry_run else 'removed'
                print(f'  [{tag}]  {stale}  (no longer assigned — deleted)')

        # Set POSIX ACLs so the user can read their directory tree via SSH/rsync.
        # This is called after copying so the ACL covers all newly created files.
        # set_acl() also grants traverse permission on all parent directories
        # above basedir — see the docstring of set_acl() for details on why this
        # is necessary on /scratch filesystems.
        if not args.email_only and not args.dry_run and found_targets:
            set_acl(basedir, user_dir, user, dry_run=False)
            print(f'  ACL set for {user}')

        if not args.email_only:
            print(f'  → {len(found_targets)} target(s), {total_files} file(s) total')
            print()

        # Build the notification email for this user (but don't send yet —
        # we collect all emails first so the send step is a separate action)
        subject, body = build_email(
            user, name, found_targets, batch_name, basedir, instrument, missing_targets,
            hostname=hostname,
        )
        email_blocks.append((user, name, info['email'], subject, body))

    # acl_pending is currently unused but kept as a hook for future deferred ACL logic
    for user, user_dir in acl_pending:
        set_acl(basedir, user_dir, user, dry_run=False)
        print(f'  ACL set for {user}')

    # Flush the checksum cache to disk so the next run can skip unchanged files
    cache.save()

    # -----------------------------------------------------------------------
    # Email summary: print all emails to stdout and write them to files
    # -----------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('EMAIL SUMMARY')
    print('=' * 60 + '\n')
    for user, name, addr, subject, body in email_blocks:
        display = f'{name} <{addr}>' if name and addr else (addr or user)
        print(f'=== To: {display} ===')
        print(f'Subject: {subject}')
        print()
        print(body)
        print()
        if not args.dry_run:
            # Save a plain-text copy of the email so the operator can review
            # or forward it manually without rerunning the script.
            email_file = os.path.join(email_dir, f'{user}_{batch_name}.txt')
            os.makedirs(email_dir, exist_ok=True)
            with open(email_file, 'w') as fh:
                fh.write(f'To: {display}\nSubject: {subject}\n\n{body}\n')
            print(f'  [email saved → {email_file}]')
        print('-' * 60)

    # --send-email: show interactive menu to choose which emails to actually send
    if args.send_email:
        interactive_send_emails(email_blocks)


if __name__ == '__main__':
    main()
