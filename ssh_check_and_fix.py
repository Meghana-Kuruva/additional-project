import paramiko, sys, time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

HOST = '61.1.175.170'; PORT = 2222
USER = 'piyush'; PASS = '2£K40d#N'
GEMINI_KEY = 'AIzaSyB0vfPqn05ZtNt2lEOgSPdvI38hyBIrW-Q'
WORK_DIR = '/home/piyush/fer_eval'
CONDA_PY = '/home/piyush/miniconda3/bin/python3'

def run(client, cmd, timeout=120):
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors='replace').strip()
    err = stderr.read().decode(errors='replace').strip()
    return out, err

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
print("Connected.\n")

# See what actually got downloaded
out, _ = run(client, f'find {WORK_DIR} -type f | sort')
print("Files in work dir:")
print(out or "(empty)")

# Look for any csv anywhere in the dir
out, _ = run(client, f'find {WORK_DIR} -name "*.csv" -o -name "*.zip" 2>/dev/null | sort')
print("\nCSV / ZIP files:")
print(out or "(none)")

# If there's a zip still around, unzip it
out, _ = run(client, f'ls {WORK_DIR}/*.zip 2>/dev/null')
if out:
    print(f"\nFound zip: {out} — unzipping...")
    out2, err2 = run(client,
        f'cd {WORK_DIR} && unzip -o *.zip 2>&1 | tail -5', timeout=120)
    print(out2 or err2)

# Now find the CSV again
out, _ = run(client, f'find {WORK_DIR} -name "*.csv" 2>/dev/null | sort')
print("\nCSV files after unzip:")
print(out or "(none)")

# If it's named differently or in a subdir, copy it to the expected location
csv_path = None
if out:
    csv_path = out.split('\n')[0].strip()
    if csv_path != f'{WORK_DIR}/fer2013.csv':
        print(f"\nCopying {csv_path} -> {WORK_DIR}/fer2013.csv")
        out2, err2 = run(client,
            f'cp "{csv_path}" {WORK_DIR}/fer2013.csv 2>&1')
        print(out2 or err2 or "OK")

# Final check
out, _ = run(client,
    f'ls -lh {WORK_DIR}/fer2013.csv 2>/dev/null && '
    f'{CONDA_PY} -c "import pandas as pd; '
    f'df=pd.read_csv(\'{WORK_DIR}/fer2013.csv\'); '
    f'print(df.shape, df.Usage.value_counts().to_dict())"')
print(f"\nDataset check:\n{out}")

# Check packages
out, _ = run(client,
    f'{CONDA_PY} -c "import google.generativeai, PIL, sklearn; print(\'packages OK\')" 2>&1')
print(f"\nPackage check: {out}")

# Upload & launch if everything is ready
if 'PrivateTest' in out or 'packages OK' in out:
    # Upload script
    EVAL_SCRIPT_PATH = f'{WORK_DIR}/eval_fer2013_gemini.py'
    sftp = client.open_sftp()
    import os
    local_script = os.path.join(
        os.path.dirname(__file__), 'eval_fer2013_gemini.py')
    sftp.put(local_script, EVAL_SCRIPT_PATH)
    sftp.close()
    print(f"\nScript uploaded -> {EVAL_SCRIPT_PATH}")

    # Launch
    log = f'{WORK_DIR}/eval_log.txt'
    launch = (
        f'cd {WORK_DIR} && GEMINI_API_KEY={GEMINI_KEY} '
        f'nohup {CONDA_PY} {EVAL_SCRIPT_PATH} > {log} 2>&1 & echo PID:$!'
    )
    out, _ = run(client, launch)
    print(f"Launched: {out}")

    time.sleep(5)
    out, _ = run(client, f'tail -10 {log}')
    print(f"\nFirst log:\n{out}")

    out, _ = run(client,
        'ps aux | grep eval_fer2013 | grep -v grep | head -2')
    print(f"\nProcess:\n{out or '(check log above)'}")
    print(f"\nMonitor: ssh piyush@61.1.175.170 -p 2222 'tail -f {log}'")
else:
    print("\nPackages not ready — install them first:")
    print(f"  ssh piyush@61.1.175.170 -p 2222")
    print(f"  {CONDA_PY} -m pip install google-generativeai pillow scikit-learn")

client.close()
