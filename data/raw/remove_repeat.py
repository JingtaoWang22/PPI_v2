f=open('yeast_ppi.tsv')
ppis=f.readlines()
f.close()



dataset=[]

for ppi in ppis:
  dataset.append(ppi.strip().split())


x=[]
y=[]

for i in range(len(dataset)):
  for j in range(i+1,len(dataset)):
    p1=dataset[i][0]
    p2=dataset[i][1]
    pa=dataset[j][0]
    pb=dataset[j][1]
    if (p1==pa and p2==pb):
      print(i,j)
      x.append(i)
      y.append(j)
    if (p1==pb and p2==pa):
      x.append(i)
      y.append(j)

y.sort()
y=y[::-1]


problems=[]

for i in y:
  problems.append(dataset[i])

for p in problems:
  dataset.remove(p)

f=open('true_yeast_ppis.tsv','w')
for d in dataset:
  f.write(str(d)+'\n')
f.close()



