import csv

#with open('eggs.csv', 'rb') as csvfile:
	#reader = csv.reader(inf)

#my_input_file = open("sarcasm_v2.csv", "r")
#my_input_file = open("test.csv", "r")
my_input_file = open("ProcessedDataset.txt", "r")

i = 1
for line in my_input_file.readlines():
	line = line.split("\t")
	if(str(line[1])== "notsarc"):
		#print(line[4])
		text_file = open("not_sarcastic_"+str(i)+".txt", "w")
		text_file.write(line[4])
		text_file.close()
		i = i +1





#text_file = open("Output.txt", "w")

#text_file.write("Purchase Amount: " 'TotalAmount')

#text_file.close()

#i =1


