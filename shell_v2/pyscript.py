import subprocess
result = subprocess.run(["ls"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print('STDOUT:', result.stdout.decode('utf-8'))
print('STDERR:', result.stderr.decode('utf-8'))
