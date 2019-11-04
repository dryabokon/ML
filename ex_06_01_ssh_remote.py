#https://habr.com/ru/post/150047/
#https://medium.com/@keagileageek/paramiko-how-to-ssh-and-file-transfers-with-python-75766179de73
# ----------------------------------------------------------------------------------------------------------------------
import paramiko
import numpy
import matplotlib.pyplot as plt

host = '192.168.1.112'
user = 'dima'
secret = 'cargo'
port = 22
# ----------------------------------------------------------------------------------------------------------------------
def example_exec_command():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, username=user, password=secret, port=port)

    stdin, stdout, stderr = client.exec_command('ls -l')
    data = stdout.read() + stderr.read()
    print(data)
    client.close()
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_sftp():

    transport = paramiko.Transport((host, port))
    transport.connect(username=user, password=secret)
    sftp = paramiko.SFTPClient.from_transport(transport)

    remotepath = '/home/dima/sources/networking/log.txt'
    localpath = './data/output/result.txt'

    sftp.get(remotepath, localpath)
    sftp.put(localpath, remotepath)

    sftp.close()
    transport.close()
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #example_sftp()

