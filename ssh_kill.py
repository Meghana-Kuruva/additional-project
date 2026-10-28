import paramiko, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('61.1.175.170', port=2222, username='piyush', password='2£K40d#N', timeout=15)
_, o, _ = client.exec_command('pkill -9 -f eval_fer2013_gemini; echo killed')
print(o.read().decode())
client.close()
