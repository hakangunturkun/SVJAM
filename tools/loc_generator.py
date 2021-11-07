import sys

old_stdout = sys.stdout
log_file = open("300k.list","w")
sys.stdout = log_file

# for i in range(85000000,95000000,300000):
for i in range(5758593,7850613,300000):
	concat_str = " chr13:"+str(i)+"-"+str(i+300000)+";chr13:"+str(i)+"-"+str(i+300000)
	print(concat_str)

for j in range(5908593,8000613,300000):
	concat_str2 = " chr13:"+str(j)+"-"+str(j+300000)+";chr13:"+str(j)+"-"+str(j+300000)
	print(concat_str2)
	  
sys.stdout = old_stdout
log_file.close()
