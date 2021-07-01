from openpyxl import load_workbook

# Load Excel File
load_wb = load_workbook("data.xlsx", data_only=True)
load_ws = load_wb['data']

# Convert Data to List Form
all_values = []
for row in load_ws.rows:
    row_value = []
    for cell in row:
        row_value.append(cell.value)
    all_values.append(row_value)

# Index List for each Time Zone of t-Day 
a = ['A','B']
b = [38]
index = 0 

for i in range(0,441):
    for k in range(0,15,2):
        p=b[index]+3
        b.append(p)
        b.append(p+1)
        index+=2
    b.pop()
    b.append(b[index-1]+65)
b.pop()

index1 = 0 
chart = [] 
for i in range(0,441):
    for j in range(0,8):
        for k in range(0,2):
            p=a[k]+str(b[index1])
            chart.append(p)
            index1+=1
            
# Input Time Zone Value
hour = input("Input Time Zone(10~17) : ")

if(hour=='10'):
    Nhour=0
elif(hour=='11'):
    Nhour=2
elif(hour=='12'):
    Nhour=4
elif(hour=='13'):
    Nhour=6
elif(hour=='14'):
    Nhour=8
elif(hour=='15'):
    Nhour=10
elif(hour=='16'):
    Nhour=12
elif(hour=='17'):
    Nhour=14

# Output for t-Day's Time Zone of Generation Quantity
line = 0 
colsum = 0 
Etotal = 0 
Hindex = 16 
length = 0 

for i in range(0,441):
    get_cells = load_ws[chart[Nhour]:chart[Nhour+1]]
    Nhour+=Hindex    
    for row in get_cells:
        colsum += row[1].value
        Etotal += row[1].value
        print("\n")
        if(line % 8==0):
            print("======================================\n")
        for cell in row:
            line+=1
            print(cell.value,"| ",end='')
            if(line % 8==0):
                print("\n\n day's",hour,"Time zone of Generation Quantity :",round(colsum,3))
                colsum = 0
    p=i+1

print("\n======================================\n")               
print("Total of Generation Quantity :",round(Etotal,3))
print("Average of Generation Quantity :", round(Etotal/p,3))
