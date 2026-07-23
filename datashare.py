"""
datashare.py — Share telluric-corrected data with collaborators.

Reads data_recipients from telluric_config.yaml, copies the relevant
tellupatched files to ./data_dir/<user>/ (or the path set by datashare_dir
in the yaml), sets ACL permissions with setfacl, writes per-user email
files, and prints a summary.

Each recipient in data_recipients may use the legacy list form (list of
targets) or the new dict form with 'email' and 'targets' keys.  If an email
address is missing the script will prompt for it and save it back to the yaml
(unless --no-prompt is given).

A CSV checksum cache (basedir/.checksum_cache.csv) stores (mtime, checksum)
per file so that FITS headers are only re-read when a file has changed,
greatly speeding up repeated runs.

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
from datetime import datetime, timezone
import shutil
import smtplib
import subprocess
import sys
from email.mime.text import MIMEText
import yaml
from tqdm import tqdm
from astropy.io import fits

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SENDER_EMAIL = 'etienne.artigau@gmail.com'


def _cfg_path(instrument='NIRPS'):
    return os.path.join(SCRIPT_DIR, f'telluric_config_{instrument.lower()}.yaml')


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def get_science_targets(config):
    """Return the sorted unique list of science targets from all data_recipients.

    Replaces the legacy top-level 'science_targets' key in the yaml.
    """
    raw = config.get('data_recipients', {})
    targets = set()
    for value in raw.values():
        if isinstance(value, list):
            targets.update(value)
        elif isinstance(value, dict):
            targets.update(value.get('targets', []))
    return sorted(targets)


def load_config(instrument='NIRPS'):
    with open(_cfg_path(instrument)) as fh:
        return yaml.safe_load(fh)


def save_config(config, instrument='NIRPS'):
    """Write the config back to telluric_config_{instrument}.yaml (comments are not preserved)."""
    with open(_cfg_path(instrument), 'w') as fh:
        yaml.dump(config, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)


def parse_recipients(raw_recipients):
    """Normalise data_recipients to {user: {'name': str|None, 'email': str|None, 'targets': list}}.

    Accepts both the legacy list form and the new dict form with name/email/targets keys.
    """
    result = {}
    for user, value in raw_recipients.items():
        if isinstance(value, list):
            result[user] = {'name': None, 'email': None, 'targets': value}
        else:
            result[user] = {
                'name': value.get('name') or None,
                'email': value.get('email') or None,
                'targets': value.get('targets', []),
            }
    return result


def prompt_missing_info(parsed, config, no_prompt=False, instrument='NIRPS'):
    """Interactively ask for missing name/email and persist to the yaml."""
    cfg_recipients = config.setdefault('data_recipients', {})
    changed = False
    for user, info in parsed.items():
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
        save_config(config, instrument=instrument)
        print(f'  [INFO] Contact info saved to telluric_config_{instrument.lower()}.yaml')


def get_gmail_app_password():
    """Retrieve the Gmail App Password from the macOS Keychain or ~/.gmail_app_password."""
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
        pass
    pw_file = os.path.expanduser('~/.gmail_app_password')
    if os.path.exists(pw_file):
        with open(pw_file) as fh:
            return fh.read().strip()
    return None


def send_gmail(recipient_email, subject, body):
    """Send an email via Gmail SMTP. Returns True on success, False on failure."""
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


EMAIL_LOG_PATH = os.path.join(SCRIPT_DIR, 'email_log.json')


def _load_email_log():
    if os.path.exists(EMAIL_LOG_PATH):
        try:
            with open(EMAIL_LOG_PATH) as fh:
                return json.load(fh)
        except Exception:
            pass
    return {}


def _save_email_log(log):
    with open(EMAIL_LOG_PATH, 'w') as fh:
        json.dump(log, fh, indent=2)


def interactive_send_emails(email_blocks):
    """Show a numbered menu of recipients and let the user choose whom to email."""
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
            email_log[user] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    _save_email_log(email_log)


def get_machine_cfg(config):
    """Return the config block for the current machine, or {}."""
    machines = config.get('machines', {})
    for _name, mcfg in machines.items():
        detect = mcfg.get('detect_path', '')
        if detect and os.path.exists(detect):
            return mcfg
    return {}


def get_project_path(config):
    return get_machine_cfg(config).get('project_path', SCRIPT_DIR)


def get_hostname(config):
    """Return the hostname of the current machine, from config if available."""
    return get_machine_cfg(config).get('hostname', None)


# ---------------------------------------------------------------------------
# Checksum cache
# ---------------------------------------------------------------------------

class ChecksumCache:
    """CSV-backed cache mapping file path → (mtime, checksum).

    Avoids re-reading FITS headers for files whose mtime has not changed.
    The cache is loaded once at startup and saved on demand.
    """

    def __init__(self, cache_path):
        self.path = cache_path
        self._cache = {}   # str -> (float, str|None)
        self._dirty = False
        self._load()

    def _load(self):
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
        """Return cached checksum if mtime still matches, else None."""
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
        return None  # stale — mtime changed

    def set(self, filepath, checksum):
        """Store checksum together with the file's current mtime."""
        try:
            mtime = os.path.getmtime(filepath)
        except OSError:
            return
        self._cache[filepath] = (mtime, checksum)
        self._dirty = True

    def save(self):
        """Persist cache to disk (no-op if nothing changed)."""
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
    """Resolve the on-disk folder for a target, trying {target}_{batch_name}_smart first,
    then {target}_{batch_name}, then plain {target}."""
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
    """Return list of files to share for a given target."""
    target_dir = find_target_dir(tellupatched_dir, target, batch_name)
    if target_dir is None:
        return []
    files = []
    for root, _dirs, fnames in os.walk(target_dir):
        for fname in fnames:
            files.append(os.path.join(root, fname))
    return sorted(files)


def fits_checksum(path, cache=None):
    """Return the CHECKSUM value from the primary HDU header.

    If *cache* is provided, return the cached value when the file mtime
    has not changed, and store newly computed values back into the cache.
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

    Copies if:
    - dst does not exist, or
    - either file lacks a FITS CHECKSUM keyword, or
    - the CHECKSUM values differ.
    """
    if not os.path.exists(dst):
        return True
    src_cs = fits_checksum(src, cache)
    dst_cs = fits_checksum(dst, cache)
    if src_cs is None or dst_cs is None:
        return True
    return src_cs != dst_cs


def copy_target(src_dir, target, dest_user_dir, dry_run, batch_name, cache=None):
    """Copy all files for a target into dest_user_dir/target/."""
    src_target = find_target_dir(src_dir, target, batch_name)
    dst_target = os.path.join(dest_user_dir, target)
    if src_target is None:
        return 0, False   # (n_files, found)

    files = collect_target_files(src_dir, target, batch_name)
    n_total = len(files)
    if not dry_run:
        os.makedirs(dst_target, exist_ok=True)
        n_copied = 0
        n_skipped = 0
        with tqdm(files, desc=target, unit='file', leave=False) as pbar:
            for fpath in pbar:
                rel = os.path.relpath(fpath, src_target)
                dst = os.path.join(dst_target, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if needs_copy(fpath, dst, cache):
                    shutil.copy2(fpath, dst)
                    if cache is not None:
                        fits_checksum(dst, cache)
                    n_copied += 1
                else:
                    n_skipped += 1
                pbar.set_postfix(copied=n_copied, skip=n_skipped)
    return n_total, True


def prune_user_dir(user_dir, expected_targets, dry_run):
    """Remove subdirectories in user_dir that are not in expected_targets.

    Returns a list of removed (or would-be-removed) directory names.
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
    """Set setfacl permissions so the user can read their folder."""
    if dry_run:
        return
    for cmd in [
        ['setfacl', '-m', f'u:{user}:r-x', basedir],
        ['setfacl', '-R', '-m', f'u:{user}:r-x', user_dir],
    ]:
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            print(f'  [WARN] setfacl not found — skipping ACL for {user}')
            break
        except subprocess.CalledProcessError as exc:
            print(f'  [WARN] setfacl failed for {user}: {exc}')
            break


def build_email(user, name, targets, batch_name, basedir, instrument, missing, hostname=None):
    """Return a (subject, body) tuple for a recipient."""
    user_path = os.path.join(basedir, user)
    if hostname:
        rsync_cmd = f'rsync -avz {user}@{hostname}:{user_path}/ ./data_from_fir/'
    else:
        rsync_cmd = f'rsync -avz <server>:{user_path}/ ./data_from_fir/'

    # Polite greeting using first name when available
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
    """Build a recipients table string for the argparse epilog."""
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

    instrument = (args.instrument or 'NIRPS').upper()
    config = load_config(instrument)

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

    project_path = get_project_path(config)
    hostname = get_hostname(config)
    batch_name = config.get('batch', {}).get('name', 'unknown')

    raw_recipients = config.get('data_recipients', {})
    if not raw_recipients:
        print(f'No data_recipients defined in telluric_config_{instrument.lower()}.yaml. Nothing to do.')
        sys.exit(0)

    if args.user:
        if args.user not in raw_recipients:
            print(f'Error: user "{args.user}" not found in data_recipients.')
            sys.exit(1)
        raw_recipients = {args.user: raw_recipients[args.user]}

    recipients = parse_recipients(raw_recipients)
    prompt_missing_info(recipients, config, no_prompt=args.no_prompt, instrument=instrument)

    tellupatched_dir = os.path.join(project_path, f'tellupatched_{instrument}')
    machine_cfg = get_machine_cfg(config)
    basedir = machine_cfg.get('datashare_dir',
                config.get('datashare_dir', os.path.join(SCRIPT_DIR, 'data_dir')))

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

    cache_path = os.path.join(basedir, '.checksum_cache.csv')
    cache = ChecksumCache(cache_path)

    email_dir = os.path.join(basedir, 'emails')
    if not args.dry_run and not args.email_only:
        os.makedirs(email_dir, exist_ok=True)

    email_blocks = []
    acl_pending = []  # [(user, user_dir)] — applied after all copies to avoid mask interference

    for user, info in sorted(recipients.items()):
        user_dir = os.path.join(basedir, user)
        targets = info['targets']
        name = info['name']
        found_targets = []
        missing_targets = []
        total_files = 0

        if not args.email_only:
            label = f'{name} ({user})' if name else user
            print(f'[{label}]')

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

        # Remove stale target directories no longer assigned to this user
        if not args.email_only:
            pruned = prune_user_dir(user_dir, set(targets), args.dry_run)
            for stale in pruned:
                tag = '(dry)' if args.dry_run else 'removed'
                print(f'  [{tag}]  {stale}  (no longer assigned — deleted)')

        if not args.email_only and not args.dry_run and found_targets:
            set_acl(basedir, user_dir, user, dry_run=False)
            print(f'  ACL set for {user}')

        if not args.email_only:
            print(f'  → {len(found_targets)} target(s), {total_files} file(s) total')
            print()

        subject, body = build_email(
            user, name, found_targets, batch_name, basedir, instrument, missing_targets,
            hostname=hostname,
        )
        email_blocks.append((user, name, info['email'], subject, body))

    # Apply ACLs after all copies are done to avoid mask interference between users
    for user, user_dir in acl_pending:
        set_acl(basedir, user_dir, user, dry_run=False)
        print(f'  ACL set for {user}')

    # Save checksum cache
    cache.save()

    # Write email files and print summary
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
            email_file = os.path.join(email_dir, f'{user}_{batch_name}.txt')
            os.makedirs(email_dir, exist_ok=True)
            with open(email_file, 'w') as fh:
                fh.write(f'To: {display}\nSubject: {subject}\n\n{body}\n')
            print(f'  [email saved → {email_file}]')
        print('-' * 60)

    if args.send_email:
        interactive_send_emails(email_blocks)


if __name__ == '__main__':
    main()
