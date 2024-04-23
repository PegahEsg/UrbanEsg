from platypus import NSGAII, Problem, Real,Permutation,Subset,CompoundOperator,SSX,SBX
import joblib
import math
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
from plotly import graph_objects as go
from shapely import geometry
from shapely.geometry import Polygon ,Point,LineString
import plotly.express as px


@st.cache_resource  
def load_model():
    model1=joblib.load('catboost.json')
    return model1
@st.cache_resource  
def load_shading_model():
    model1=joblib.load('catboost_shading.json')
    return model1

model = load_model()
model2=load_shading_model()

BuildingShape1=st.sidebar.radio("Building Shape",["Boxy","L-shape"])
if BuildingShape1=="Boxy":
    BuildingShape=0
else:
    BuildingShape=1
green_ratio=st.sidebar.slider("Green space ratio",min_value=0.3 ,max_value=0.7,step=0.1)
SiteLength=st.sidebar.slider("SiteLength",min_value=100,max_value=150,step=5)
Site_Width=100
u=st.sidebar.slider("U",min_value=2,max_value=4,step=1)
v=st.sidebar.slider("V",min_value=2,max_value=6,step=1)
n_parcel=u*v
Densityeachbldg=st.sidebar.slider("Density each building",min_value=181,max_value=600,step=10)
st.sidebar.header('Energy')
Heating=st.sidebar.checkbox('Heating')
if Heating:
    Heating_up = st.sidebar.slider("Heating",100,1000,50)

Cooling=st.sidebar.checkbox('Cooling')
if Cooling:
    Cooling_up = st.sidebar.slider("Cooling",100,1000,50)

Lighting=st.sidebar.checkbox('Lighting')
if Lighting:
    Lighting_up = st.sidebar.slider("Lighting",100,1000,50)
Gas=st.sidebar.checkbox('Gas')
if Gas:
    GAS_up = st.sidebar.slider("GAS",100,1000,50)

st.sidebar.header('Environment')
Hours=st.sidebar.checkbox('SolarHours')
if Hours:
    Hours_up = st.sidebar.slider("Hours",100,1000,50)
CO2=st.sidebar.checkbox('CO2 Emission')
if CO2:
    CO2_up = st.sidebar.slider("CO2",100,1000,500)
Shade=st.sidebar.checkbox('Shaded Area')
if Shade:
    Shade_up = st.sidebar.slider("Shader",1,100,50)
Radiation=st.sidebar.checkbox('Radiation')
if Radiation:
    Radiation_up = st.sidebar.slider("Radiation",100,1000,50)
st.sidebar.header('Livability')
SVF=st.sidebar.checkbox('SVF')
if SVF:
    max_svf=st.sidebar.checkbox('Maximum')
    min_svf=st.sidebar.checkbox('Minimum')
Visibility=st.sidebar.checkbox('Visibility')
if Visibility:
    Visibility_down = st.sidebar.slider("Visibility",100,1000,50) 
st.sidebar.header('Renewable')
PV=st.sidebar.checkbox('PV')
if PV:
    PV_down = st.sidebar.slider("PV",100,1000,50)

true_indexes=[]
my_list=[Heating,Cooling,Lighting,Gas,Radiation,Hours,CO2,SVF,Visibility,PV]
for i in range(len(my_list)):
    if my_list[i] == True:
        true_indexes.append(i)

options=int(st.sidebar.number_input("how many Alternative do you want?"))
on = st.sidebar.button('Optimize')  
def neigbors_h(stories,park_loc,u,v):
    def add_padding(matrix):
        padded_matrix = [[0] * (len(matrix[0]) + 2)]  # Adding top padding
        for row in matrix:
            padded_row = [0] + row + [0]
            padded_matrix.append(padded_row)
        padded_matrix.append([0] * (len(matrix[0]) + 2))  # Adding bottom padding
        return padded_matrix

    def get_neighbors(matrix, row, col):
        neighbors = [
            matrix[row-1][col-1], matrix[row-1][col], matrix[row-1][col+1],  # North-West, North, North-East
            matrix[row][col-1], matrix[row][col+1],                        # West, East
            matrix[row+1][col-1], matrix[row+1][col], matrix[row+1][col+1]   # South-West, South, South-East
        ]
        return neighbors

    rounded_means = [round((a + b) / 2) for a, b in zip(stories[0::2], stories[1::2])]
    new_stories=rounded_means.copy()
    for l in park_loc:
        new_stories.insert(l,0)
    new_stories= np.reshape(new_stories,[u,v]).tolist()
    padded_matrix = add_padding(new_stories)
    height=[]
    for i in range(1, len(padded_matrix)-1):
        for j in range(1, len(padded_matrix[0])-1):
            current_element = padded_matrix[i][j]
            neighbors = get_neighbors(padded_matrix, i, j)
            neighbors.insert(4,current_element)
            height.append([neighbors[1],neighbors[2],neighbors[5],neighbors[8],neighbors[7],neighbors[6],neighbors[3],neighbors[0]]) 
    return height
def create_polygon(width, length, centroid):
    half_width = width / 2
    half_length = length / 2
    
    vertices = [
        (centroid[0] - half_length, centroid[1] - half_width),
        (centroid[0] + half_length, centroid[1] - half_width),
        (centroid[0] + half_length, centroid[1] + half_width),
        (centroid[0] - half_length, centroid[1] + half_width)
    ]

    return vertices

def adjacency_estimation(stories):
    adjacency=[]

    for i,j in zip(stories,range(0,len(stories))):
        if j%2==0:
            adjacency.append([0,stories[j+1]])
        else:
            adjacency.append([stories[j-1],0])
    return adjacency
    
def create_l_shape(count,width, length, centroid):   
    haf_w = width / 2
    haf_l = length / 2
    x=centroid[0]
    y=centroid[1]   
    if count%2==0:
        vertices=([(x , y),(x+haf_l,y),(x+haf_l,y-haf_w),(x-haf_l,y-haf_w),(x-haf_l,y+haf_w),(x,y+haf_w)])           
    else:
        vertices=([(x , y),(x-haf_l,y),(x-haf_l,y-haf_w),(x+haf_l,y-haf_w),(x+haf_l,y+haf_w),(x,y+haf_w)])    
    return vertices

def shading (BuildingShape,orientation,building_parcels_loc,Lengths,Widths,SiteLength,Site_Width,zz):
    x1=[]
    y1=[]

    for i in building_parcels_loc:
        x1.append(i[0])
        y1.append(i[1])
    xx =[]
    for i in x1:
        if (100/u) < (SiteLength/v):
            xx.append(i-(0)-Widths/2)
            xx.append(i+(0)+Widths/2)
        else:
            xx.append(i-(0)-Lengths/2)
            xx.append(i+(0)+Lengths/2)

    yy = []
    for i in y1:
        yy.append(i)
        yy.append(i)

    xy=[[x,y] for x,y in zip(xx,yy)]

    P1=geometry.Polygon([(-1,-1),(Site_Width+1,-1),(Site_Width+1,SiteLength+1),(-1,SiteLength+1)])

    if BuildingShape==0:
        for i in xy:
            width =Widths 
            length = Lengths
            coordinate=i
            centroid = (coordinate[0], coordinate[1])  
            vertices = create_polygon(width, length, coordinate)
            P2=geometry.Polygon(vertices)
            P1=P1.difference(P2)
    else:
        count=0
        for i in xy:
            width =Widths 
            length = Lengths
            centroid = (i[0], i[1])  
            vertices = create_l_shape(count,width, length, centroid)
            P2=geometry.Polygon(vertices)
            P1=P1.difference(P2)
            count+=1

    polygan=P1
    latmin, lonmin, latmax, lonmax =polygan.bounds
    valid_points=[]

    xx_p,yy_p = np.meshgrid(np.arange(latmin, latmax,1),np.arange(lonmin, lonmax,1))

    pts = [Point(X,Y) for X,Y in zip(xx_p.ravel(),yy_p.ravel())]

    points = [pt for pt in pts if pt.within(polygan)]
    x_p=[]
    y_p=[]
    z1=len(points)*[0]
    for i in range(0,len(points)):
        x_p.append(points[i].x)
        y_p.append(points[i].y)  
    d=[]
    for i,j in zip(x_p,y_p):
        d.append([i,j])

    po=pd.DataFrame(d,columns=['x','y'])
    sto=zz #stories

    shading_pro=[]
    p=0
    l=0
    for i in d[:]:
        y_neigbor=[]
        k=po.query(f'x=={i[0]}')
        for j in range(1,SiteLength):
            if j in k.y.values:
                y_neigbor.append(1)
            else:
                y_neigbor.append(0)
        if 0 in y_neigbor:
            s=pd.DataFrame({'all':list(range(1,SiteLength)),'yn':y_neigbor})
            try:
                s1=s.query(f'yn==0 and all>={i[1]}')
                s2=s1.sort_values(by='all')
                s3=s2['all'].head(1).values[0]
                sou_dis=s3-i[1]
                distance_building_center=[]
                for dis in xy: 
                    distance_building_center.append(abs(i[0]-dis[0])+abs(i[1]-dis[1]))
                stories=sto
                hei_shading=pd.DataFrame({'stories':stories,'dis':distance_building_center})
                hei_shading1=hei_shading.sort_values(by='dis')['stories'].head(1).values
                shading_pro.append(model2.predict_proba([[BuildingShape,orientation,sou_dis,hei_shading1[0]]])[0][0])
                l=l+1
            except:
                    shading_pro.append(1)
                    o=p+1
        else:
            shading_pro.append(1)

            p=p+1
    st.write(l)
    return shading_pro,x_p,y_p

def towers(a, e, pos_x, pos_y,res_com):
# create points
    x, y, z = np.meshgrid(
        np.linspace(pos_x-a/2, pos_x+a/2, 2), 
        np.linspace(pos_y-a/2, pos_y+a/2, 2), 
        np.linspace(0, e, 2)
    )
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    print(x,y,z,"flatten")
    if res_com==0:
        return  go.Mesh3d(x=x, y=y, z=z, alphahull=1, flatshading=True,color='wheat')
    else :
        return  go.Mesh3d(x=x, y=y, z=z, alphahull=1, flatshading=True,color='gray')

def comm(a, e, pos_x, pos_y,cm):
# create points
    x, y, z = np.meshgrid(
        np.linspace(pos_x-a/2, pos_x+a/2, 2), 
        np.linspace(pos_y-a/2, pos_y+a/2, 2), 
        np.linspace(cm*3.1, e, 2)
    )
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    
    if res_com==0:
        return  go.Mesh3d(x=x, y=y, z=z, alphahull=1, flatshading=True,color='wheat')
    else:
        return  go.Mesh3d(x=x, y=y, z=z, alphahull=1, flatshading=True,color='gray')



def park(a, e, pos_x, pos_y):
# create points
    x, y, z = np.meshgrid(
        np.linspace(pos_x-a/2, pos_x+a/2, 2), 
        np.linspace(pos_y-a/2, pos_y+a/2, 2), 
        np.linspace(0, e, 2)
    )
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    
    return go.Mesh3d(x=x, y=y, z=z, alphahull=1, flatshading=True,color='limegreen')


def n_park_f(x):
    return round(x * green_ratio)

outputs=[]	
def predictive_model(BuildingShape,green_ratio,SiteLength,Rotation,Sub_street,Bldg_Footprint,Densityeachbldg,
                     WWR,Lengths,Widths,u,v,stories,Area,adjacency,height,dimention,res_ratio,
                     com_ratio,com_floor,res_com_loc,building_loc_copy):
    e_h=[]
    e_c=[]
    e_l=[]
    e_g=[]
    
    en_r=[]
    en_h=[]
    en_co=[]
    en_sha=[]
    
    l_s=[]
    l_v=[]
    
    r_pv=[]
    j=0
    for building,com_off,k in zip(range(len(building_loc_copy)),res_com_loc,building_loc_copy):
        if com_off==0:
            y=[BuildingShape,green_ratio,SiteLength,Rotation,Sub_street,Bldg_Footprint,Densityeachbldg,WWR,
               Lengths,Widths,u,v,stories[building],Area,0,adjacency[building][0],adjacency[building][1],
               *height[k],*dimention[k],res_ratio,com_ratio] #res

        else:     
            y=[BuildingShape,green_ratio,SiteLength,Rotation,Sub_street,Bldg_Footprint,Densityeachbldg,WWR,
               Lengths,Widths,u,v,stories[building],Area,com_floor[j],adjacency[building][0],adjacency[building][1],
               *height[k],*dimention[k],res_ratio,com_ratio]#com_off
            j+=1
        predict=model.predict(y)
         
        r_pv.append(predict[0])
        
        e_c.append(predict[1])
        e_h.append(predict[2])
        e_l.append(predict[3])
        e_g.append(predict[4])
        
        
        en_h.append(predict[5])
        en_r.append(predict[6])
        en_co.append(predict[7])
        #en_sha.append(shading(BuildingShape,var[0][0],building_coor,Lengths,Widths,SiteLength,Site_Width,zz))
        
        l_s.append(predict[8])
        l_v.append(predict[9])
    return r_pv,e_c,e_h,e_l,e_g,en_h,en_r,en_co,l_s,l_v
def Area_building(BuildingShape,Lengths,Widths):
    if BuildingShape==0:
        Area=Lengths*Widths
    else:
        Area=(Lengths*(Widths*2)*0.75)
    return Area
def optimize(input):
    Rotation = input[0][0]
    Sub_street = input[1][0]
    Bldg_Footprint = input[2][0]
    WWR = input[3][0]
    com_floor = input[11][0]
    building_loc=input[10]
   
    s = building_loc
    park_loc = [x for x in list(range(n_parcel)) if x not in s]

    res_ratio=input[12][0]
    com_ratio=input[15][0]
    
    if res_ratio==0.5:
        res_com_loc=input[13]
        com_floor=input[11]
    else:
        res_com_loc=input[14]
        com_floor=input[16]
    
    print(res_com_loc,com_floor)
    
    if 180<=Densityeachbldg<=240:
        if Bldg_Footprint==0.45:
            stories=input[4]
        else:
            stories=input[4]

    elif 240<Densityeachbldg<=300:
        if Bldg_Footprint==0.45:
            stories=input[5]
        else:
            stories=input[4]           
    elif 300<Densityeachbldg<=360:
        if Bldg_Footprint==0.45:
            stories=input[7]
        else:
            stories=input[5]

    elif 360<Densityeachbldg<=420:
        if Bldg_Footprint==0.45:
            stories=input[8]
        else:
            stories=input[6]
    elif 420<Densityeachbldg<=480:
        if Bldg_Footprint==0.45:
            stories=input[9]
        else:
            stories=input[7]

    elif 480<Densityeachbldg<=540:
        stories=input[8]

    elif 540<Densityeachbldg<=600:
        stories=input[9]

    pairs = [(stories[i], stories[i + 1]) for i in range(0, len(stories), 2)]
    unpacked_list = [element for pair in pairs for element in pair[::-1]]

    adjacency=adjacency_estimation(stories)
    
    Lengths = ((SiteLength/v)-Sub_street)/2
    Widths = ((100/u)-Sub_street)*Bldg_Footprint

    Area=Area_building(BuildingShape,Lengths,Widths)
    #Dens = (sum(Stories)*Lengths*Widths)/(140*90)
    height=neigbors_h(stories,park_loc,u,v,)        
    j=0
    building_loc_copy=building_loc.copy()*2
         
    r_pv,e_c,e_h,e_l,e_g,en_h,en_r,en_co,l_s,l_v=predictive_model(BuildingShape,green_ratio,SiteLength,Rotation,Sub_street,Bldg_Footprint,Densityeachbldg,WWR,Lengths,Widths,u,v,stories,Area,adjacency,height,dimention,res_ratio,com_ratio,com_floor,res_com_loc,building_loc_copy)
        
    Objectives=[]   
    if Heating:
        if sum(e_h)<Heating_up:
            Objectives.append(sum(e_h))
        else:
            Objectives.append(sum(e_h)*10000000)
            
    if Cooling:
        if sum(e_h)<Cooling_up:
            Objectives.append(sum(e_h))
        else:
            Objectives.append(sum(e_c)*10000000)
            
    if Lighting:
        if sum(e_l)<Lighting_up:
            Objectives.append(sum(e_l))
        else:
            Objectives.append(sum(e_l)*10000000)      
    if Gas:
        if sum(e_g)<GAS_up:
            Objectives.append(sum(e_g))
        else:
            Objectives.append(sum(e_g)*10000000)  
            
    if Radiation:
        if sum(en_r)<Radiation_up:
            Objectives.append(sum(en_r))
        else:
            Objectives.append(sum(en_r)*10000000)  
    if Hours:
        if sum(en_h)<Hours_up:
            Objectives.append(sum(en_h))
        else:
            Objectives.append(sum(en_h)*10000000)  
    if CO2:
        if sum(en_co)<CO2_up:
            co2=(((sum(e_l)+sum(e_c))*0.21233/1000)+((sum(e_h)+sum(e_g))*0.18316/1000))
            Objectives.append(co2)
        else:
            Objectives.append(sum(en_co)*10000000)  
    if SVF:
        if max_svf:
            Objectives.append(-1*sum(l_s))
        else:
            Objectives.append(sum(l_s))  
    if Visibility:
        if Visibility_down<sum(l_v):
            Objectives.append(sum(l_s))
        else:
            Objectives.append(sum(l_s)*10000000)          
    if PV:
        if sum(r_pv)>PV_down:
            Objectives.append(sum(r_pv))  
        else:
            Objectives.append(sum(r_pv)*10000000)            
                
    return Objectives 
    
def X_var():
    
    
    var0 = Subset([0,45,90],1) #Rotation
    var1 = Subset([5,10],1) #Sub Street
    var2 = Subset([0.45,0.6],1) #Building Footprint
    var3 = Subset([1,2],1) #WWR
           
    
    
    var4 = Subset(2*(n_parcel - round(n_parcel * green_ratio))*[4,5],(n_parcel - round(n_parcel * green_ratio))*2) #Stories
    var5 = Subset(2*(n_parcel - round(n_parcel * green_ratio))*[4,5,6],(n_parcel - round(n_parcel * green_ratio))*2) #Stories
    var6 = Subset(2*(n_parcel - round(n_parcel * green_ratio))*[5,6,7],(n_parcel - round(n_parcel * green_ratio))*2) #Stories
    var7 = Subset(2*(n_parcel - round(n_parcel * green_ratio))*[6,7,8],(n_parcel - round(n_parcel * green_ratio))*2) #Stories
    var8 = Subset(2*(n_parcel - round(n_parcel * green_ratio))*[7,8,9],(n_parcel - round(n_parcel * green_ratio))*2) #Stories
    var9 = Subset(2*(n_parcel - round(n_parcel * green_ratio))*[8,9,10],(n_parcel - round(n_parcel * green_ratio))*2) #Stories
    
    var10 = Subset(list(range(n_parcel)),n_parcel - round(n_parcel * green_ratio))
    var11=Subset([1,2,3]*len(one_nim)*2,len(one_nim)*2) #Com_floor_vaghti nesfe mahale edarie
    var12=Subset([0.5,0.7],1) #res_ratio

    var13=Permutation((zero_nim+one_nim))
    var14=Permutation((zero_haf+one_haf)*2)
    
    var15=Subset([0.1,0.2,0.3],1) #com_ratio
    var16=Subset([1,2,3]*len(one_nim)*2,len(one_haf)*2) #Com_floor_vaghti 0.3 mahale edarie
    return var0,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10,var11,var12,var13,var14,var15,var16
        
if on:    
    n_park = n_park_f(n_parcel)
    parcel_list = list(range(n_parcel))
    park_parcels = np.sort(np.random.choice(parcel_list, size=n_park, replace=False))
    building_parcels = [i for i in parcel_list if i not in park_parcels]

    parcels_X = []
    for i in range(u):
        parcels_X.append(((100/u/2)*(i*2+1)))

    parcels_Y = []
    for i in range(v):
        parcels_Y.append(((SiteLength/v/2)*(i*2+1)))

    parcels_loc=[]
    for i in parcels_Y:
        for j in parcels_X:
            parcels_loc.append([j,i])

    zero_nim=[0]*round(((n_parcel - round(n_parcel * green_ratio))*0.5)*2)
    one_nim=[1]*round(((n_parcel - round(n_parcel * green_ratio))*0.5)*2)

    zero_haf=[0]*round((n_parcel - round(n_parcel * green_ratio))*0.7)*2
    one_haf=[1]*round((n_parcel - round(n_parcel * green_ratio))*0.3)*2

    width=np.linspace(0,SiteLength,v+1)
    c_w=[]
    for i,j in zip(width,width[1:]):
        c_w.append(np.mean([i,j]))

    length=np.linspace(0,100,u+1)
    l_w=[]
    for i,j in zip(length,length[1:]):
        l_w.append(np.mean([i,j]))

    coor_parcel=[]
    for i in c_w:
        for j in l_w:
            coor_parcel.append([i,j])
            
    s=[300,*[math.dist(coor_parcel[0],coor_parcel[1])]*(u-1)]*v
    n=[*[math.dist(coor_parcel[0],coor_parcel[1])]*(u-1),300]*v
    w=[*[300]*u,*[math.dist(coor_parcel[0],coor_parcel[u])]*(len(coor_parcel)-u)]
    e=[*[math.dist(coor_parcel[0],coor_parcel[u])]*(len(coor_parcel)-u),*[300]*u]
    nw=[*[300]*u,*[*[math.sqrt((math.dist(coor_parcel[0],coor_parcel[u])**2)+(math.dist(coor_parcel[0],coor_parcel[1])**2))]*(u-1),300]*(v-1)]
    ne=[*[*[math.sqrt((math.dist(coor_parcel[0],coor_parcel[u])**2)+(math.dist(coor_parcel[0],coor_parcel[1])**2))]*(u-1),300]*(v-1),*[300]*u]
    sw=[*[300]*u,*[300,*[math.sqrt((math.dist(coor_parcel[0],coor_parcel[u])**2)+(math.dist(coor_parcel[0],coor_parcel[1])**2))]*(u-1)]*(v-1)]
    se=[*[300,*[math.sqrt((math.dist(coor_parcel[0],coor_parcel[u])**2)+(math.dist(coor_parcel[0],coor_parcel[1])**2))]*(u-1),300]*(v-1),*[300]*u]

    dimention=[]
    for dN,dNE,dE,dSE,dS,dSW,dW,dNW in zip(n,ne,e,se,s,sw,w,nw):
        dimention.append([dN,dNE,dE,dSE,dS,dSW,dW,dNW])


    problem = Problem(17, len(true_indexes))
    problem.types[:] = X_var()
    problem.function = optimize
    algorithm = NSGAII(problem, variator=CompoundOperator(SSX(),SBX()))
    algorithm.run(1)


    data=pd.DataFrame()
    col=st.columns(options)
    opt=1
    for solution in algorithm.result[0:options]:
        st.header(f"Alternative {opt}")
        opt+=1
        var=solution.variables
        OB=[]
        OB_name=[]
        data1=pd.DataFrame([[var[0][0],var[1][0],var[2][0],var[3][0],var[12][0],var[15][0]]],columns=['rotation', 'Sub_Street', 'Bldgfootprint', 'WWR', 'res_ratio','com_ratio'])
        Lengths = ((SiteLength/v)-var[1][0])/2
        Widths = ((100/u)-var[1][0])*var[2][0]
        Bldg_Footprint=var[2][0]
        if 180<Densityeachbldg<=240:
            if Bldg_Footprint==0.45:
                stories=var[4]
            else:
                stories=var[4]

        elif 240<Densityeachbldg<=300:
            if Bldg_Footprint==0.45:
                stories=var[5]
            else:
                stories=var[4]           
        elif 300<Densityeachbldg<=360:
            if Bldg_Footprint==0.45:
                stories=var[7]
            else:
                stories=var[5]

        elif 360<Densityeachbldg<=420:
            if Bldg_Footprint==0.45:
                stories=var[8]
            else:
                stories=var[6]
        elif 420<Densityeachbldg<=480:
            if Bldg_Footprint==0.45:
                stories=var[9]
            else:
                stories=var[7]

        elif 480<Densityeachbldg<=540:
            stories=var[8]

        elif 540<Densityeachbldg<=600:
            stories=var[9]
        data2=pd.DataFrame([stories],columns=['h'+str(i) for i in range(len(stories))])
        data3=pd.DataFrame([var[10]],columns=['parcel_building'+str(i) for i in range(0,len(var[10]))])

        res_ratio=var[12][0]
        com_ratio=var[15][0]

        if res_ratio==0.5:
            res_com_loc=var[13]
            com_floor=var[11]
        else:
            res_com_loc=var[14]
            com_floor=var[16]

        data4=pd.DataFrame([res_com_loc],columns=['com or res?'+str(i) for i in range(0,len(res_com_loc))])
        data5=pd.DataFrame([com_floor],columns=['com floor'+str(i) for i in range(0,len(com_floor))])
        my_list1=["Heating","Cooling","Lighting","Gas","Radiation","Hours","CO2","Shade","SVF","Visibility","PV"]
        
        tr=0
        for i in true_indexes:
            OB.append(solution.objectives[tr])
            OB_name.append(my_list1[i])
            tr+=1

        data6=pd.DataFrame([OB],columns=OB_name)
        result = pd.concat([data1, data2, data3, data4, data5,data6], axis=1)
        
        data=pd.concat([data,result])   
        building_loc=var[10]
        if var[12][0]==0.5:
            res_com_loc=var[13]
            com_floor=var[11]
        else:
            res_com_loc=var[14]
            com_floor=var[16]
            
        res_com=res_com_loc
        commercial=[]
        cm=0
        for i in res_com:
            if i==0:
                commercial.append(0)
            else:
                commercial.append(com_floor[cm])
                cm+=1
        building_coor=[]
        for i in building_loc:
            building_coor.append(parcels_loc[i])
        
        s = set(building_loc)
        park_loc = [x for x in list(range(n_parcel)) if x not in s]
        park_coor=[]
        for i in park_loc:
            park_coor.append(parcels_loc[i])

        x1=[]
        y1=[]
        
        for i in building_coor:
            x1.append(i[0])
            y1.append(i[1])
        x2=[]
        y2=[]
        for i in park_coor:
            x2.append(i[0])
            y2.append(i[1])
        xx=[]
        for i in x1:
            if (100/u) < (SiteLength/v):
                xx.append(i-(0)-Widths/2)
                xx.append(i+(0)+Widths/2)
            else:
                xx.append(i-(0)-Lengths/2)
                xx.append(i+(0)+Lengths/2)
        yy = []
        for i in y1:
            yy.append(i)
            yy.append(i)
        zz = [k*3.1 for k in stories]
        
        xx1=[]
        for i in x2:
            xx1.append(i)
            xx1.append(i+9)
        yy1=[]
        for i in y2:
            yy1.append(i)
            yy1.append(i)
        zz1=len(y2)*[0.2]+len(y2)*[0.2]
        
        xx2 =[]
        for i in x1:
            if (100/u) < (SiteLength/v):
                xx2.append(i-(0)-Widths/2)
                xx2.append(i+(0)+Widths/2)
            else:
                xx2.append(i-(0)-Lengths/2)
                xx2.append(i+(0)+Lengths/2)
           
        yy2 = yy
        zz2 = commercial
                       
        Area=Area_building(BuildingShape,Lengths,Widths)
        adjacency=adjacency_estimation(zz)
        height=neigbors_h(zz,park_loc,u,v) #### 
        building_loc_copy=building_loc.copy()*2
        
        sum_res=[]
        sum_com=[]
        for s,rc in zip(zz,res_com_loc):
            if rc==0:
                sum_res.append(s)
            else:
                sum_com.append(s)
        nafar_res1=round((sum(sum_res)*Area)/35)
        nafar_office=round((sum(sum_com)*Area)/9.3)
        nafar_comm=round((len(com_floor)*Area)/6.2)    
        Machine_res=round(nafar_res1/4)
        Machine_office=round(nafar_office/2)
        Machine_comm=round(nafar_comm/2)   
        lowest_length=Widths+Lengths
        longest_length=max((u*Widths)+Lengths,(v*Lengths)+Widths)

        st.write(f"in this generated option the number of residential occupants is {nafar_res1} with {Machine_res} cars,the number of office occupants is {nafar_office} with {Machine_office} cars, and number of commercial occupants is {nafar_comm} with {Machine_comm}. In this option the longest way is {longest_length} and shortest length is {lowest_length} ")
    
        
        r_pv,e_c,e_h,e_l,e_g,en_h,en_r,en_co,l_s,l_v=predictive_model(BuildingShape,green_ratio,SiteLength,var[0][0],var[1][0],var[2][0],
                                                                      Densityeachbldg,var[3][0],Lengths,Widths,u,v,zz,Area,adjacency,
                                                                      height,dimention,var[12][0],var[15][0],com_floor,res_com_loc,building_loc_copy)


        fig = go.Figure(layout={'scene': {'aspectmode':"data"}})
        fig.update_layout(
            scene = dict(
                xaxis = dict(visible=False),
                yaxis = dict(visible=False),
                zaxis =dict(visible=True,backgroundcolor="whitesmoke")
                )
            )
        for x, y, z,l in zip(xx, yy, zz,res_com,):
            fig.add_trace(towers(9, z, x, y,l))

        for i,j,k in zip(xx1,yy1,zz1):
            fig.add_trace(park(12, k, i,j))#
            
        for x, y, z,l in zip(xx2,yy2,zz2,commercial):
            fig.add_trace(comm(9.05, z, x, y,l))#
        st.markdown("3D view")
        st.plotly_chart(fig)
        #st.write([len(i)for i in [zz,height,dimention,res_com_loc,building_loc_copy]])
        #st.write(res_com_loc,res_ratio)
        each=pd.DataFrame({"stories":zz,'r_pv':r_pv,"e_c":e_c,"e_h":e_h,"e_l":e_l,"e_g":e_g,"en_h":en_h,"en_r":en_r,"en_co":en_co,"l_s":l_s,"l_v":l_v})
        st.markdown("Outputs for each building")
        st.dataframe(each)
        
        if Shade:
            shading1=shading(BuildingShape,var[0][0],building_coor,Lengths,Widths,SiteLength,Site_Width,zz)
            po=pd.DataFrame({"x":shading1[1],"y":shading1[2],"shading":shading1[0]})
            po['z']=1     
            plt.scatter(po['x'],po['y'],c=po['shading'],s=1)
            st.markdown("Shaded Area")
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            #fig = go.Figure(data=[go.Scatter3d(x=po['x'],y=po['y'],z=po['z'],mode='markers',marker=dict(size=0.0,color=po['shading']))])
            #fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        
            if (sum(shading1[0])/len(shading1[0]))>(1-(Shade/100)):
                st.write("AVE>User")
                
            else:
                st.write("no option")


        
            st.dataframe(data)
            outputs.append(algorithm.result)
            st.divider()
else:
    st.write("tap toggle to optimize ")
    
    
    

        
        



    
    
