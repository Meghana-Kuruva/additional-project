import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('61.1.175.170', port=2222, username='piyush', password='2£K40d#N', timeout=15)

cmds = [
    'ls ~/BTP/',
    'find ~/BTP -name "*.csv" 2>/dev/null | head -10',
    'find ~/BTP -name "fer*" 2>/dev/null | head -10',
    'which python3',
    'pip3 list 2>/dev/null | grep -i "kaggle\|generative"',
    'ls ~/.kaggle/kaggle.json 2>/dev/null || echo no-kaggle-creds',
    'conda env list 2>/dev/null | head -10',
]

for cmd in cmds:
    _, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode(errors='replace').strip()
    err = stderr.read().decode(errors='replace').strip()
    print(f'--- {cmd}')
    print(out or err or '(empty)')
    print()

client.close()
