"""
datashare.py — Share telluric-corrected data with collaborators.

Reads data_recipients from telluric_config.yaml, copies the relevant
tellupatched files to /scratch/eartigau/datashare/<user>/, sets ACL
permissions with setfacl, and prints a summary of emails to send.

Usage
-----
    python datashare.py                     # copy + set ACL + print emails
    python datashare.py --dry-run           # preview without copying
    python datashare.py --user alexsm       # only process one recipient
    python datashare.py --email-only        # only print email summaries
    python datashare.py --instrument SPIROU # override instrument
"""

import argparse
import os
import shutil
import subprocess
import sys
import yaml
from astropy.io import fits

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config():
    cfg_path = os.path.join(SCRIPT_DIR, 'telluric_config.yaml')
    with open(cfg_path) as fh:
        return yaml.safe_load(fh)


def get_project_path(config):
    machines = config.get('machines', {})
    for _name, mcfg in machines.items():
        detect = mcfg.get('detect_path', '')
        if detect and os.path.exists(detect):
            return mcfg['project_path']
    return SCRIPT_DIR


def get_hostname(config):
    """Return the hostname of the current machine, from config if available."""
    machines = config.get('machines', {})
    for _name, mcfg in machines.items():
        detect = mcfg.get('detect_path', '')
        if detect and os.path.exists(detect):
            return mcfg.get('hostname', None)
    return None


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


def fits_checksum(path):
    """Return the CHECKSUM value from the primary HDU header, or None if absent/unreadable."""
    try:
        hdr = fits.getheader(path, ext=0)
        return hdr.get('CHECKSUM', None)
    except Exception:
        return None


def needs_copy(src, dst):
    """Return True if src should be copied to dst.

    Copies if:
    - dst does not exist, or
    - either file lacks a FITS CHECKSUM keyword, or
    - the CHECKSUM values differ.
    """
    if not os.path.exists(dst):
        return True
    src_cs = fits_checksum(src)
    dst_cs = fits_checksum(dst)
    if src_cs is None or dst_cs is None:
        return True
    return src_cs != dst_cs


def copy_target(src_dir, target, dest_user_dir, dry_run, batch_name):
    """Copy all files for a target into dest_user_dir/target/."""
    src_target = find_target_dir(src_dir, target, batch_name)
    dst_target = os.path.join(dest_user_dir, target)
    if src_target is None:
        return 0, False   # (n_files, found)

    files = collect_target_files(src_dir, target, batch_name)
    n_total = len(files)
    if not dry_run:
        os.makedirs(dst_target, exist_ok=True)
        for i, fpath in enumerate(files, 1):
            rel = os.path.relpath(fpath, src_target)
            dst = os.path.join(dst_target, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if needs_copy(fpath, dst):
                print(f'    [{i}/{n_total}] {rel}', flush=True)
                shutil.copy2(fpath, dst)
            else:
                print(f'    [{i}/{n_total}] {rel}  [skip — checksum match]', flush=True)
    return n_total, True


def set_acl(basedir, user_dir, user, dry_run):
    """Set setfacl permissions so the user can read their folder."""
    if dry_run:
        return
    for cmd in [
        ['setfacl', '-m', f'u:{user}:rx', basedir],
        ['setfacl', '-R', '-m', f'u:{user}:rx', user_dir],
    ]:
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            print(f'  [WARN] setfacl not found — skipping ACL for {user}')
            break
        except subprocess.CalledProcessError as exc:
            print(f'  [WARN] setfacl failed for {user}: {exc}')
            break


def build_email(user, targets, batch_name, basedir, instrument, missing, hostname=None):
    """Return a formatted email string for a recipient."""
    user_path = os.path.join(basedir, user)
    if hostname:
        rsync_cmd = f'rsync -avz {user}@{hostname}:{user_path}/ ./{user}/'
    else:
        rsync_cmd = f'rsync -avz <server>:{user_path}/ ./{user}/'
    lines = [
        f'=== Email to {user} ===',
        f'Subject: {instrument} telluric-corrected data available (batch {batch_name})',
        '',
        f'Hi {user},',
        '',
        f'Your telluric-corrected {instrument} data for batch {batch_name} is now available at:',
        f'  {user_path}/',
        '',
        'To download your data, run the following command on your local machine:',
        f'  {rsync_cmd}',
        '',
    ]

    if targets:
        lines.append(f'Targets ({len(targets)}):')
        for t in targets:
            lines.append(f'  - {t}')
    else:
        lines.append('No targets were found on disk for you in this batch.')

    if missing:
        lines.append('')
        lines.append(f'Note: the following targets were attributed to you but had no data on disk:')
        for t in missing:
            lines.append(f'  - {t}')

    lines += [
        '',
        'Best,',
        'Étienne',
        '',
        '-' * 60,
    ]
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Share telluric-corrected data with collaborators.',
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
    args = parser.parse_args()

    config = load_config()
    instrument = (args.instrument or config.get('instrument', 'NIRPS')).upper()
    project_path = get_project_path(config)
    hostname = get_hostname(config)
    batch_name = config.get('batch', {}).get('name', 'unknown')

    recipients = config.get('data_recipients', {})
    if not recipients:
        print('No data_recipients defined in telluric_config.yaml. Nothing to do.')
        sys.exit(0)

    if args.user:
        if args.user not in recipients:
            print(f'Error: user "{args.user}" not found in data_recipients.')
            sys.exit(1)
        recipients = {args.user: recipients[args.user]}

    tellupatched_dir = os.path.join(project_path, f'tellupatched_{instrument}')
    basedir = '/scratch/eartigau/datashare'

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

    email_blocks = []
    acl_pending = []  # [(user, user_dir)] — applied after all copies to avoid mask interference

    for user, targets in sorted(recipients.items()):
        user_dir = os.path.join(basedir, user)
        found_targets = []
        missing_targets = []
        total_files = 0

        if not args.email_only:
            print(f'[{user}]')

        for target in targets:
            n_files, found = copy_target(tellupatched_dir, target, user_dir, args.dry_run or args.email_only, batch_name)
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

        if not args.email_only and not args.dry_run and found_targets:
            set_acl(basedir, user_dir, user, dry_run=False)
            print(f'  ACL set for {user}')

        if not args.email_only:
            print(f'  → {len(found_targets)} target(s), {total_files} file(s) total')
            print()

        email_blocks.append(build_email(
            user, found_targets, batch_name, basedir, instrument, missing_targets,
            hostname=hostname,
        ))

    # Apply ACLs after all copies are done to avoid mask interference between users
    for user, user_dir in acl_pending:
        set_acl(basedir, user_dir, user, dry_run=False)
        print(f'  ACL set for {user}')

    # Print email summary
    print('\n' + '=' * 60)
    print('EMAIL SUMMARY')
    print('=' * 60 + '\n')
    for block in email_blocks:
        print(block)


if __name__ == '__main__':
    main()
