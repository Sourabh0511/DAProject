import csv


my_input_file = open("ProcessedDataset.txt", "r")

i = 1
for line in my_input_file.readlines():
	line = line.split("\t")
	if(str(line[1])== "sarc"): #replace sarc with nonsarc for the other set
		#print(line[4])
		text_file = open("sarcastic_"+str(i)+".txt", "w")
		text_file.write(line[4])
		text_file.close()
		i = i +1








