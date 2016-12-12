import pandas

filename = "GadgetManiacs_Cluster.xlsx"
#filename = "Supermarket.xlsx"

dataDf = pandas.read_excel(filename)
data = dataDf.values.tolist()
lines = len(data)
columns = len(data[0])
print("\nNumber of lines:", lines)
print("Number of columns:", columns)
