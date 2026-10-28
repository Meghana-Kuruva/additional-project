import paramiko, sys, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

HOST='61.1.175.170'; PORT=2222; USER='piyush'; PASS='2£K40d#N'
GEMINI_KEY = 'AIzaSyB0vfPqn05ZtNt2lEOgSPdvI38hyBIrW-Q'
WORK_DIR='/home/piyush/fer_eval'
CONDA_PY='/home/piyush/miniconda3/bin/python3'
LOG=f'{WORK_DIR}/eval_log.txt'

def run(client, cmd, timeout=30):
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors='replace').strip()
    err = stderr.read().decode(errors='replace').strip()
    return out, err

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

# Check if eval is running
out, _ = run(client, 'ps aux | grep eval_fer2013 | grep -v grep')
print("Running processes:")
print(out or "(none — not running)")

# Check log
out, _ = run(client, f'cat {LOG} 2>/dev/null | tail -20')
print(f"\nLog tail:\n{out or '(empty)'}")

# If not running, launch it with increased timeout on nohup
if not out or 'eval_fer2013' not in out:
    print("\nNot running — launching now...")
    launch = (
        f'cd {WORK_DIR} && '
        f'GEMINI_API_KEY={GEMINI_KEY} '
        f'nohup {CONDA_PY} {WORK_DIR}/eval_fer2013_gemini.py '
        f'>> {LOG} 2>&1 &'
    )
    # Use get_pty to avoid hang
    chan = client.get_transport().open_session()
    chan.exec_command(launch)
    time.sleep(3)
    chan.close()
    print("Launch command sent.")

    time.sleep(4)
    out, _ = run(client, 'ps aux | grep eval_fer2013 | grep -v grep | head -2')
    print(f"Process check:\n{out or '(still not running — see log)'}")

    out, _ = run(client, f'tail -10 {LOG}')
    print(f"\nLog:\n{out}")

client.close()
