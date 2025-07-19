from platypus import NSGAII, Problem, Real,Permutation,Subset,CompoundOperator,SSX,SBX
import joblib
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
import io
import random

buffer = io.BytesIO()
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://images.unsplash.com/photo-1566041510639-8d95a2490bfb?q=80&w=1956&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;}
</style>
"""
st.markdown(background_image, unsafe_allow_html=True)
@st.cache_resource  
def load_model_sv():
    model1=joblib.load('s_v.json')
    return model1
@st.cache_resource  
def load_model():
    model1=joblib.load('xgboost(2).json')
    return model1
@st.cache_resource  
def load_shading_model():
    model1=joblib.load('catboost_shading.json')
    return model1

model = load_model()
model1=load_model_sv()
model2=load_shading_model()

st.sidebar.header('Define Site Parameters')
BuildingShape1=st.sidebar.radio("Building Shape",["Rectangle","L-shape"])
if BuildingShape1=="Rectangle":
    BuildingShape=0
else:
    BuildingShape=1
green_ratio=st.sidebar.slider("Green Space Ratio",min_value=0.3 ,max_value=0.7,step=0.1)
SiteLength=st.sidebar.slider("Site Length (m)",min_value=100,max_value=150,step=5)
Site_Width=100
u=st.sidebar.slider("Number of Parcel in a Row",min_value=2,max_value=4,step=1)
v=st.sidebar.slider("Number of Parcels in a Column",min_value=2,max_value=6,step=1)

if v < u:
    st.sidebar.warning("⚠️ For better results, consider setting the number of parcels in a column greater than or equal to the number in a row.")
  
n_parcel=u*v
Densityeachbldg=st.sidebar.slider("Building Density (%)",min_value=180,max_value=600,step=15)


ad=st.sidebar.toggle('Advance Mode', value=False, disabled=False, label_visibility="visible")
if ad==True:
    gas_car_rate=st.sidebar.slider("Gasoline Car Ratio",min_value=0.10,max_value=1.0,step=0.05) #####
    st.sidebar.header('Energy Metrics')
    Heating=st.sidebar.checkbox('Heating (kWh/m2)')
    if Heating:
        Heating1 = st.sidebar.select_slider("Heating",options=['Low','Medium','High'])
        with st.sidebar.expander("ℹ️"):
            st.write(" 40<Low<60 , 60<Medium<80 , 80<High<100 ")
        if Heating1=='Low':
            Heating_down=40
            Heating_up=60
        elif Heating1=='Medium':
            Heating_down=60
            Heating_up=80
        else:
            Heating_down=80
            Heating_up=100
            
    Cooling=st.sidebar.checkbox('Cooling (kWh/m2)')
    
    if Cooling:
        Cooling1 =  st.sidebar.select_slider("Cooling",options=['Low','Medium','High'])
        with st.sidebar.expander("ℹ️"):
            st.write(" 70<Low<100 , 100<Medium<130 , 130<High<160 ")
        if Cooling1=='Low':
            Cooling_down=70
            Cooling_up=100
        elif Cooling=='Medium':
            Cooling_down=100
            Cooling_up=130
        else:
            Cooling_down=130
            Cooling_up=160
    
    Lighting=st.sidebar.checkbox('Lighting (kWh/m2)')
    if Lighting:
        Lighting1= st.sidebar.select_slider("Lighting",options=['Low','Medium','High'])
        with st.sidebar.expander("ℹ️"):
            st.write(" 15<Low<25 , 25<Medium<32 , 32<High<40 ")
        if Lighting1=='Low':
            Lighting_down=15
            Lighting_up=25
        elif Lighting1=='Medium':
            Lighting_down=25
            Lighting_up=32
        else:
            Lighting_down=32
            Lighting_up=40
    st.sidebar.header('Environmental Metrics')
    
    Co2=st.sidebar.checkbox('Co2 Emission-kg')
    if Co2:
        Co2_up = st.sidebar.slider("Co2",5,200,5)
    
    
    Hours=st.sidebar.checkbox('Solar Hours (Hr)')
    if Hours:
        Hours_up = st.sidebar.slider("Hours",min_value=1.0,max_value=5.0,step=0.5)
    
    roof_cold=st.sidebar.checkbox('Radiation (Coldest Week)-kWh/m2')
    if roof_cold:
        roof_cold_up = st.sidebar.slider("Roof Cold up",50,1000,50)
    
    Shade=st.sidebar.checkbox('Shaded Area (%)')
    
    if Shade:
        
        shading = st.sidebar.select_slider("Shading",
        options=["Low","Moderate","High","Very High", "Maximum"],
        value=("Moderate"))
        with st.sidebar.expander("ℹ️"):
            st.write(" Low=20 , Moderate=40 , High=60 , Very High=80 , Maximum=100 ")
        if shading=="Low":
            Shade_up=20
        elif shading=="Moderate":
            Shade_up=40
        elif shading=="High":
            Shade_up=60
        elif shading=="Very High":
            Shade_up=90
        elif shading=="Maximum":
            Shade_up=100
    
    roof_hot=st.sidebar.checkbox('Radiation (Hottest Week)-kWh/m2')
    if roof_hot:
        roof_hot_up = st.sidebar.slider("Roof Hot up",100,1000,50)
    st.sidebar.header('Livability Metrics')
    SVF=st.sidebar.checkbox('Sky View Factor (%)')
    if SVF:
        SVF1= st.sidebar.select_slider("Sky View Factor",options=['Limited sky view','Moderate sky view','Extensive sky view'])
        with st.sidebar.expander("ℹ️"):
            st.write(" 30<Limited sky view<50 , 50<Moderate<75 , 75<Extensive sky view<100 ")
        if SVF1=='Limited sky view':
            SVF_down=30
            SVF_up=50
        elif SVF1=='Moderate sky view':
            SVF_down=50
            SVF_up=75
        else:
            SVF_down=75
            SVF_up=100
    Visibility=st.sidebar.checkbox('Park Visibility (%)')
    if Visibility:
        Visibility1=st.sidebar.select_slider("Visibility",options=['Obstructed view','Partial view','Clear view']) 
        with st.sidebar.expander("ℹ️"):
            st.write(" 30<Obstructed view<50 , 50<Partial view<75 , 75<Clear view<100 ")
        if Visibility1=='Obstructed view':
            Visibility_down=30
            Visibility_up=50
        elif Visibility1=='Partial view':
            Visibility_down=50
            Visibility_up=75
        else:
            Visibility_down=75
            Visibility_up=100
    
    st.sidebar.header('Renewable')
    PV=st.sidebar.checkbox('PV Power Generation (%)')
    if PV:
        #PV_down = st.sidebar.slider("PV",100,1000,50)
        PV_down = st.sidebar.slider("What percentage of energy is supplied by PV?",0,50,2)
        PV_down=PV_down/100
    EUI=False
else:
    gas_car_rate=1
    Heating=False
    EUI=st.sidebar.checkbox('Energy Use Intensity (kWh/m2)')
    if EUI:
        EUI1 = st.sidebar.select_slider("EUI",options=['Uncertified','EC','EC+','EC++'])
        with st.sidebar.expander("ℹ️"):
            st.write(" 80<EC++<130 , 130<EC+<180 , 180<EC<290 , Uncertified>290 ")
        if EUI1=='EC':
            EUI_down=180
            EUI_up=290
        elif EUI1=='EC+':
            EUI_down=130
            EUI_up=180
        elif EUI1=='EC++' :
            EUI_down=80
            EUI_up=130
        else:
            EUI_down=291
            EUI_up=10000
            
        
            
    Cooling=False 
    Lighting=False
    st.sidebar.header('Environmental Metrics')
    
    Co2=False
    #Co2_up=50
    Hours=st.sidebar.checkbox('Solar Hours (Hr)')
    if Hours:
        Hours_up = st.sidebar.slider("Hours",min_value=1.0,max_value=5.0,step=0.5)

    roof_cold=False
    #roof_cold_up=500
    
    Shade=st.sidebar.checkbox('Shaded Area %')
    if Shade:
        shading = st.sidebar.select_slider("Shading",
        options=["Low","Moderate","High","Very high", "Maximum"],
        value=("Moderate"))
        with st.sidebar.expander("ℹ️"):
            st.write(" Low=20 , Moderate=40 , High=60 , Very High=80 , Maximum=100 ")
        if shading=="Low":
            Shade_up=20
        elif shading=="Moderate":
            Shade_up=40
        elif shading=="High":
            Shade_up=60
        elif shading=="Very high":
            Shade_up=90
        elif shading=="Maximum":
            Shade_up=100

    roof_hot=False
    #roof_hot_up=500
    st.sidebar.header('Livability Metrics')
    SVF=st.sidebar.checkbox('Sky View Factor (%)')
    if SVF:
        SVF1= st.sidebar.select_slider("Sky View Factor",options=['Limited sky view','Moderate sky view','Extensive sky view'])
        with st.sidebar.expander("ℹ️"):
            st.write(" 30<Limited sky view<50 , 50<Moderate<75 , 75<Extensive sky view<100 ")
        if SVF1=='Limited sky view':
            SVF_down=30
            SVF_up=50
        elif SVF1=='Moderate sky view':
            SVF_down=50
            SVF_up=75
        else:
            SVF_down=75
            SVF_up=100
    Visibility=st.sidebar.checkbox('Park Visibility (%)')
    if Visibility:
        Visibility1=st.sidebar.select_slider("Visibility",options=['Obstructed view','Partial view','Clear view']) 
        with st.sidebar.expander("ℹ️"):
            st.write(" 30<Obstructed view<50 , 50<Partial view<75 , 75<Clear view<100 ")
        if Visibility1=='Obstructed view':
            Visibility_down=30
            Visibility_up=50
        elif Visibility1=='Partial view':
            Visibility_down=50
            Visibility_up=75
        else:
            Visibility_down=75
            Visibility_up=100
    
    st.sidebar.header('Renewable')
    PV=st.sidebar.checkbox('PV Power Generation (%)')
    if PV:
        #PV_down = st.sidebar.slider("PV",100,1000,50)
        PV_down = st.sidebar.slider("What percentage of energy is supplied by PV?",0,50,step=2)
        PV_down=PV_down/100
        

true_indexes=[]
my_list=[EUI,Cooling,Heating,Lighting,roof_hot,Hours,roof_cold,SVF,Visibility,PV,Co2]
for i in range(len(my_list)):
    if my_list[i] == True:
        true_indexes.append(i)

options=int(st.sidebar.number_input("How many Alternative do you want?",min_value=1,max_value=5,value=1,step=1))
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
            length = Lengths*0.9
            coordinate=i
            centroid = (coordinate[0], coordinate[1])  
            vertices = create_polygon(width, length, coordinate)
            P2=geometry.Polygon(vertices)
            P1=P1.difference(P2)
    else:
        count=0
        for i in xy:
            width =Widths
            length = Lengths*0.9
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
        np.linspace(pos_x-a[0]/2, pos_x+a[0]/2, 2), 
        np.linspace(pos_y-a[1]/2, pos_y+a[1]/2, 2), 
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
        np.linspace(pos_x-a[0]/2, pos_x+a[0]/2, 2), 
        np.linspace(pos_y-a[1]/2, pos_y+a[1]/2, 2), 
        np.linspace(cm*3.5, e, 2)
    )
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    
    if res_com==0:
        return  go.Mesh3d(x=x, y=y, z=z, alphahull=1, flatshading=True,color='wheat')
    else:
        return  go.Mesh3d(x=x, y=y, z=z, alphahull=1, flatshading=True,color='gray')

def park(a, e, pos_x, pos_y):
    

    x, y, z = np.meshgrid(
        np.linspace(pos_x-a/2, pos_x+a/2, 2), 
        np.linspace(pos_y-a/2, pos_y+a/2, 2), 
        np.linspace(0, e, 2))
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    
    return go.Mesh3d(x=x, y=y, z=z, alphahull=1, flatshading=True,color='limegreen')
    
def centerpark(a, e, pos_x, pos_y,name):
    x, y, z = np.meshgrid(
        np.linspace(pos_x - a / 2, pos_x + a / 2, 1), 
        np.linspace(pos_y - a / 2, pos_y + a / 2, 1), 
        np.linspace(e, e, 1)
    )
    
    # Flatten the arrays to create a list of coordinates
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    return go.Scatter3d(x=x, y=y, z=z,textfont=dict(size=12,color='black'),showlegend=False, mode='markers+text',text=[name],textposition="top center",marker=dict(size=3,color='black'))

def north(a, e, pos_x, pos_y,name):
    x, y, z = np.meshgrid(
        np.linspace(pos_x - a / 2, pos_x + a / 2, 1), 
        np.linspace(pos_y - a / 2, pos_y + a / 2, 1), 
        np.linspace(0, e, 1)
    )
    
    # Flatten the arrays to create a list of coordinates
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    return go.Scatter3d(x=x, y=y, z=z,textfont=dict(size=20,color='black'),showlegend=False, mode='markers+text',text=[name],textposition="top center",marker=dict(size=3,color='black'))

def n_park_f(x):
    return round(x * green_ratio)

outputs=[]	
def predictive_model(BuildingShape,green_ratio,SiteLength,Rotation,Sub_street,Bldg_Footprint,Densityeachbldg,
                     WWR,Lengths,Widths,u,v,stories,Area,adjacency,height,dimention,res_ratio,
                     com_ratio,com_floor,res_com_loc,building_loc_copy):
    e_h=[]
    e_c=[]
    e_l=[]

    
    en_r=[]
    en_h=[]
    en_co=[]
    en_sha=[]
    hours=[]
    l_s=[]
    l_v=[]
    
    r_pv=[]                    
    j=0
    for building,com_off,k in zip(range(len(building_loc_copy)),res_com_loc,building_loc_copy):
        if com_off==0:
            y=[BuildingShape,green_ratio,SiteLength,Rotation,Sub_street,Bldg_Footprint,Densityeachbldg,1,
               Lengths,Widths,u,v,stories[building],Area,0,adjacency[building][0],adjacency[building][1],
               *height[k],*dimention[k],res_ratio,com_ratio] #res

        else:     
            y=[BuildingShape,green_ratio,SiteLength,Rotation,Sub_street,Bldg_Footprint,Densityeachbldg,2,
               Lengths,Widths,u,v,stories[building],Area,com_floor[j],adjacency[building][0],adjacency[building][1],
               *height[k],*dimention[k],res_ratio,com_ratio]#com_off
            j+=1
        predict=model.predict(y)
        #predict=[round(i,2) for i in predict]
        r_pv.append(predict[0])
        
        e_c.append(predict[1])
        e_h.append(predict[2])
        e_l.append(predict[3])

        
        
        en_h.append(predict[5])
        hours.append(predict[6])
        en_co.append(predict[7])
        #en_sha.append(shading(BuildingShape,var[0][0],building_coor,Lengths,Widths,SiteLength,Site_Width,zz))

        if predict[8]<0:
            l_s.append(random.randint(20,30))
        elif predict[8]>100:
            l_s.append(random.randint(70,90))
        else:
            l_s.append(abs(predict[8]))
            
        l_v.append(0)
    return r_pv,e_c,e_h,e_l,en_h,hours,en_co,l_s,l_v


def predictive_model_park(BuildingShape,green_ratio,SiteLength,Rotation,Sub_street,Bldg_Footprint,Lengths,Widths,u,v,Area,height,dimention,park_loc):

    e_h=[]
    e_c=[]
    e_l=[]
   
    en_r=[]
    en_h=[]
    en_co=[]
    en_sha=[]
    hours=[]
    l_s=[]
    l_v=[]
    
    r_pv=[]  
                   
    j=0
    
    for k in park_loc:
        
        y=[BuildingShape,green_ratio,SiteLength,Rotation,Sub_street,Bldg_Footprint,Lengths,Widths,u,v,Area,*height[k],*dimention[k]] #res
        
        predict=model1.predict(y)
        #predict=[round(i,2) for i in predict]
        e_c.append(0)
        e_h.append(0)
        e_l.append(0)
    
        en_h.append(0)
        hours.append(0)
        en_co.append(0)  

        if predict[0]<0:
            l_s.append(random.randint(20,30))
        elif predict[0]>100:
            l_s.append(random.randint(70,90))
        else:
            l_s.append(abs(predict[0]))

        if predict[1]*2.5 <=100:
            l_v.append(predict[1]*2.5)
        else:
             l_v.append(100)

    
    return r_pv,e_c,e_h,e_l,en_h,hours,en_co,l_s,l_v


def Area_building(BuildingShape,Lengths,Widths):
    if BuildingShape==0:
        Area=Lengths*Widths
    else:
        Area=(Lengths*(Widths)*0.75) #######
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

    sum_res=[]
    sum_com=[]
    for rc in zip(res_com_loc):
        if rc==0:
            sum_res.append(1)
        else:
            sum_com.append(1)
    
    
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

    elif 540<Densityeachbldg<=700:
        stories=input[9]

    pairs = [(stories[i], stories[i + 1]) for i in range(0, len(stories), 2)]
    unpacked_list = [element for pair in pairs for element in pair[::-1]]

    adjacency=adjacency_estimation(stories)

    #124
    Lengths = ((SiteLength-((v+1)*Sub_street))/v)/2
    Widths = ((100-(u+1)*Sub_street)/u)*Bldg_Footprint

    

    Area=Area_building(BuildingShape,Lengths,Widths)
    #Dens = (sum(Stories)*Lengths*Widths)/(140*90)
    height=neigbors_h(stories,park_loc,u,v,)        
    j=0
    building_loc_copy=building_loc.copy()*2
    
    nafar_res1=round((sum(sum_res)*Area)/17) #####
    nafar_office=round(((sum(sum_com)*com_ratio)*Area)/9.3)
    nafar_comm=round(((sum(sum_com)*(1-com_ratio))*Area)/6.2)
    Machine_res=round(nafar_res1/4)
    Machine_office=round(nafar_office/2)
    Machine_comm=round(nafar_comm/2)   
    lowest_length=round(Widths+Lengths,2)
    longest_length=round(max((u*Widths)+Lengths,(v*Lengths)+Widths),2)

    num_gas_car=int((Machine_res+Machine_office+Machine_comm)*gas_car_rate) #####
    num_cng_car=int((Machine_res+Machine_office+Machine_comm)*(1-gas_car_rate))###


    total_co2_gas=round(0.07956*num_gas_car*2.31*longest_length,2)/1000 #####
    total_co2_CNG=round(0.06453*num_gas_car* 2.75*longest_length,2)/1000 #####
    total_co2=round(total_co2_gas+total_co2_CNG,2) #####


    r_pv_building,e_c_building,e_h_building,e_l_building,en_h_building,hours_building,en_co_building,l_s_building,l_v_building=predictive_model(BuildingShape,green_ratio,SiteLength,Rotation,Sub_street,Bldg_Footprint,Densityeachbldg,WWR,Lengths,Widths,u,v,stories,Area,adjacency,height,dimention,res_ratio,com_ratio,com_floor,res_com_loc,building_loc_copy)
    r_pv_park,e_c_park,e_h_park,e_l_park,en_h_park,hours_park,en_co_park,l_s_park,l_v_park=predictive_model_park(BuildingShape,green_ratio,SiteLength,Rotation,Sub_street,Bldg_Footprint,Lengths,Widths,u,v,Area,height,dimention,park_loc)

    
    
    Objectives=[]

    if EUI:
        EUI_C=np.mean(e_h_building)*1.1+(np.mean(e_c_building)+np.mean(e_l_building))*3.1
        if EUI_down<EUI_C<EUI_up:
            Objectives.append(np.mean(EUI_C))
        else:
            Objectives.append(EUI_C*10000000)
    
    if Heating:
        if Heating_down<np.mean(e_h_building)<Heating_up:
            Objectives.append(np.mean(e_h_building))
        else:
            Objectives.append(np.mean(e_h_building)*10000000)
            
    if Cooling:
        if Cooling_down<np.mean(e_c_building)<Cooling_up:
            Objectives.append(np.mean(e_c_building))
        else:
            Objectives.append(np.mean(e_c_building)*10000000)
            
    if Lighting:
        if Lighting_down<Lighting_down<np.mean(e_l_building)<Lighting_up:
            Objectives.append(np.mean(e_l_building))
        else:
            Objectives.append(np.mean(e_l_building)*10000000)      
            
    if roof_hot:
        if np.mean(en_h_building)<roof_hot_up:
            Objectives.append(np.mean(en_h_building))
        else:
            Objectives.append(np.mean(en_h_building)*10000000)  
    if Hours:
        if np.mean(hours_building)<Hours_up:
            Objectives.append(np.mean(hours_building))
        else:
            Objectives.append(np.mean(hours_building)*10000000)  
    if roof_cold:
        if np.mean(en_co_building)<roof_cold_up:
            Objectives.append(np.mean(en_co_building))
        else:
            Objectives.append(np.mean(en_co_building)*10000000)  
             
    if SVF:
        if SVF_down<np.mean(l_s_building+l_s_park)<SVF_up:
            Objectives.append(np.mean(l_s_building+l_s_park))
        else:
            Objectives.append(np.mean(l_s_building+l_s_park)*10000000)  
    if Visibility:
        if Visibility_down<np.mean(l_v_park)<Visibility_up:
            Objectives.append(np.mean(l_v_park))
        else:
            Objectives.append(np.mean(l_v_park)*0.000001)          
    if PV:
        Total_E=np.array(e_l_building)+np.array(e_c_building)+np.array(e_h_building)
        Total_E_stories=Total_E.dot(stories)
        
        if PV_down-0.05< np.mean(r_pv_building)/np.mean(Total_E_stories)< PV_down+0.05:
            Objectives.append(np.mean(r_pv_building)/np.mean(Total_E_stories))
            #st.write(np.mean(r_pv_building)/np.mean(Total_E_stories))
        else:
            Objectives.append((np.mean(r_pv_building)/np.mean(Total_E_stories))*10000000)            
    if Co2:
        co2_total=(sum(e_h_building)*0.21233 +sum(e_c_building)*0.18316 +sum(e_l_building)*0.21233)*Area+ total_co2
        if  co2_total<Co2_up:
            Objectives.append(co2_total)
        else:
            Objectives.append(co2_total*10000000)       
    return Objectives 
    
def X_var():
    
    
    var0 = Subset([-45,0,45],1) #Rotation
    var1 = Subset([6,12],1) #Street width
    var2 = Subset([0.45,0.6],1) #Building Footprint
    var3 = Subset([1,1],1) #WWR
           
    
    
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
    var14=Permutation((zero_haf+one_haf))
    
    var15=Subset([0.1,0.2,0.3],1) #com_ratio
    var16=Subset([1,2,3]*len(one_nim)*2,len(one_haf)*2) #Com_floor_vaghti 0.3 mahale edarie
    return var0,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10,var11,var12,var13,var14,var15,var16
        
if on:    
    n_park = n_park_f(n_parcel)
    parcel_list = list(range(n_parcel))
    park_parcels = np.sort(np.random.choice(parcel_list, size=n_park, replace=False))
    building_parcels = [i for i in parcel_list if i not in park_parcels]

    parcels_X = []
    #arman
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
    algorithm.run(100)


    data=pd.DataFrame()
    col=st.columns(options)
    opt=1
    download=pd.DataFrame()
    for solution in algorithm.result:
        var=solution.variables
        OB=[]
        OB_name=[]
        data_general=pd.DataFrame([[green_ratio,var[0][0],var[1][0],var[2][0],var[12][0],var[15][0]]],columns=['Green Space Ratio','Rotation (Degree)', 'Street width (m)', 'Building footprint','Residental Ratio','Commercial Ratio'])
        
        Lengths = ((SiteLength-((v+1)*var[1][0]))/v)/2
        Widths = ((100-(u+1)*var[1][0])/u)*var[2][0]
        Bldg_Footprint=var[2][0]
        if 180<=Densityeachbldg<=240:
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

        elif 540<Densityeachbldg<=700:
            stories=var[9]
        
        res_ratio=var[12][0]
        com_ratio=var[15][0]

        if res_ratio==0.5:
            res_com_loc=var[13]
            com_floor=var[11]
        else:
            res_com_loc=var[14]
            com_floor=var[16]

        my_list1=["EUI","Cooling","Heating","Lighting","Roof Hot","Hours","Roof Cold","SVF","Visibility","PV","Co2"]
        
        tr=0
        for i in true_indexes:
            OB.append(solution.objectives[tr])
            OB_name.append(my_list1[i])
            tr+=1
        
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
        #e2
        for i in x1:
            xx.append(i-(0)-Lengths/2)
            xx.append(i+(0)+Lengths/2)
        yy = []
        for i in y1:
            yy.append(i)
            yy.append(i)
        zz = [k*3.5 for k in stories]
        
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
            xx2.append(i-(0)-Lengths/2)
            xx2.append(i+(0)+Lengths/2)
           
        yy2 = yy
        zz2 = commercial
                       
        Area=Area_building(BuildingShape,Lengths,Widths)
        adjacency=adjacency_estimation(zz)
        height=neigbors_h(zz,park_loc,u,v) #### 
        building_loc_copy=building_loc.copy()*2
        Rotation=var[0][0]
        sum_res=[]
        sum_com=[]
        for s,rc in zip(zz,res_com_loc):
            if rc==0:
                sum_res.append(s/3.5)
            else:
                sum_com.append(s/3.5)
        #st.write(Area,sum_res,sum_com)
        nafar_res1=round((sum(sum_res)*Area)/17) #####
        nafar_office=round(((sum(sum_com)*var[15][0])*Area)/9.3)
        nafar_comm=round(((sum(sum_com)*(1-var[15][0]))*Area)/6.2)
        Machine_res=round(nafar_res1/4)
        Machine_office=round(nafar_office/4)
        Machine_comm=round(nafar_comm/4)   
        lowest_length=round(Widths+Lengths,2)
        longest_length=round(max((u*Widths)+Lengths,(v*Lengths)+Widths),2)

        num_gas_car=int((Machine_res+Machine_office+Machine_comm)*gas_car_rate) #####
        num_cng_car=int((Machine_res+Machine_office+Machine_comm)*(1-gas_car_rate))###


        total_co2_gas=round(0.07956*num_gas_car*2.31*longest_length,2) #####
        total_co2_CNG=round(0.06453*num_gas_car* 2.75*longest_length,2) #####
        total_co2=round(total_co2_gas+total_co2_CNG,2) #####

                
        r_pv_building,e_c_building,e_h_building,e_l_building,en_h_building,hours_building,en_co_building,l_s_building,l_v_building=predictive_model(BuildingShape,green_ratio,SiteLength,var[0][0],var[1][0],var[2][0],Densityeachbldg,var[3][0],Lengths,Widths,u,v,zz,Area,adjacency,height,dimention,var[12][0],var[15][0],com_floor,res_com_loc,building_loc_copy)
        r_pv_park,e_c_park,e_h_park,e_l_park,en_h_park,hours_park,en_co_park,l_s_park,l_v_park=predictive_model_park(BuildingShape,green_ratio,SiteLength,var[0][0],var[1][0],var[2][0],Lengths,Widths,u,v,Area,height,dimention,park_loc)


        if BuildingShape==0:
            area=[(2*i*3.5*Widths)+(2*i*3.5*Lengths)+(Site_Width*SiteLength) for i in zz]
            vol=[area*sto*3.5 for area,sto in zip(area,zz)]
        else:
            area=[(2*i*3.5*Widths)+(2*i*3.5*Lengths)+(Site_Width*SiteLength)*75 for i in zz]
            vol=[area*sto*3.5*0.75 for area,sto in zip(area,zz)]

        
        each=pd.DataFrame({"location x":xx,"location y":yy,"Height":zz,"Number of Floor":[i/3.5 for i in zz],"Aspect Ratio":[i/var[1][0] for i in zz],'PV generation (kWh/m2)':r_pv_building,"Cooling (kWh/m2)":e_c_building,"Heating (kWh/m2)":e_h_building,"Lighting (kWh/m2)":e_l_building,"Roof hot (kWh/m2)":en_h_building,"Solar Hours (Hours)":hours_building,"Roof Cold (kWh/m2)":en_co_building,"SVF %":l_s_building})
        each=pd.DataFrame({"location x":xx,"location y":yy,"Height":zz,"Number of Floor":[i/3.5 for i in zz],"Aspect Ratio":[i/var[1][0] for i in zz],'PV generation (kWh/m2)':r_pv_building,"Cooling (kWh/m2)":e_c_building,"Heating (kWh/m2)":e_h_building,"Lighting (kWh/m2)":e_l_building,"Roof hot (kWh/m2)":en_h_building,"Solar Hours (Hours)":hours_building,"Roof Cold (kWh/m2)":en_co_building,"SVF %":l_s_building})

        each['Co2']=each['Cooling (kWh/m2)']*1.1 + each['Heating (kWh/m2)']*1.1 +each['Lighting (kWh/m2)']*1.1
        each['EUI']=each['Cooling (kWh/m2)']*1.1 + each['Heating (kWh/m2)']*3.1 +each['Lighting (kWh/m2)']*1.1
        each.round({'Co2':2})
        each_parks=pd.DataFrame({'name':['park '+str(i) for i in list(range(1,len(l_s_park)+1))],"location":park_coor,"SVF %":l_s_park,"Visibility %":l_v_park})
        
        each=pd.concat([each,each_parks])
        each=pd.concat([each,data_general])
        
        download=pd.concat([download,each])
    
        
    for solution in algorithm.result[0:options]:
        st.header(f"Alternative {int(opt)}")
        opt+=1
        var=solution.variables
        OB=[]
        OB_name=[]
                
        Lengths = ((SiteLength-((v+1)*var[1][0]))/v)/2
        Widths = ((100-(u+1)*var[1][0])/u)*var[2][0]
        data1=pd.DataFrame([[round(Widths,ndigits=1),round(Lengths,ndigits=1),var[0][0],var[1][0],var[2][0],var[12][0],var[15][0]]],columns=['Width (m)','Length (m)','Rotation (Degree)', 'Street width (m)', 'Building footprint','Residental Ratio','Commercial Ratio'])

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

        elif 540<Densityeachbldg<=700:
            stories=var[9]
        #data2=pd.DataFrame([stories],columns=['h'+str(i) for i in range(len(stories))])
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
        my_list1=["Cooling","Heating","Lighting","Roof Hot","Hours","Roof Cold","SVF","Visibility","PV","Co2"]
        
        tr=0
        for i in true_indexes:
            OB.append(solution.objectives[tr])
            OB_name.append(my_list1[i])
            tr+=1

        data6=pd.DataFrame([OB],columns=OB_name)
        result = pd.concat([data1,data3, data4, data5,data6], axis=1)
        
        data=pd.concat([data,result])  

        st.dataframe(data1)
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
            xx.append(i-(0)-Lengths/2)
            xx.append(i+(0)+Lengths/2)
        yy = []
        for i in y1:
            yy.append(i)
            yy.append(i)
        zz = [k*3.5 for k in stories]
        
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
            xx2.append(i-(0)-Lengths/2)
            xx2.append(i+(0)+Lengths/2)
           
        yy2 = yy
        zz2 = commercial
                       
        Area=Area_building(BuildingShape,Lengths,Widths)
        adjacency=adjacency_estimation(zz)
        height=neigbors_h(zz,park_loc,u,v) #### 
        building_loc_copy=building_loc.copy()*2
        Rotation=var[0][0]
        sum_res=[]
        sum_com=[]
        for s,rc in zip(zz,res_com_loc):
            if rc==0:
                sum_res.append(s/3.5)
            else:
                sum_com.append(s/3.5)
        #st.write(Area,sum_res,sum_com)
        nafar_res1=round((sum(sum_res)*Area)/17.5) #####
        nafar_office=round(((sum(sum_com)*var[15][0])*Area)/9.3)
        nafar_comm=round(((sum(sum_com)*(1-var[15][0]))*Area)/6.2)

        pc_res=(sum(sum_res)*Area)/nafar_res1
        pc_comm=round(((sum(sum_com)*(1-var[15][0]))*Area))/nafar_comm
        pc_office=round(((sum(sum_com)*var[15][0])*Area))/nafar_office

        #parka kam beshe
        pc_out=((Site_Width*SiteLength)-(Widths*Lengths*u*v))/(nafar_res1+nafar_comm+nafar_office)
        
        Machine_res=round(nafar_res1/4)
        Machine_office=round(nafar_office/2)
        Machine_comm=round(nafar_comm/2)   

        
        lowest_length=round(Widths+Lengths,2)
        longest_length=round(max((u*Widths)+Lengths,(v*Lengths)+Widths),2)

        num_gas_car=int((Machine_res+Machine_office+Machine_comm)*gas_car_rate) #####
        num_cng_car=int((Machine_res+Machine_office+Machine_comm)*(1-gas_car_rate))###


        total_co2_gas=round(0.07956*num_gas_car*2.31*longest_length,2) #####
        total_co2_CNG=round(0.06453*num_gas_car* 2.75*longest_length,2) #####
        total_co2=round(total_co2_gas+total_co2_CNG,2) #####

        st.markdown(f"""
            - Residential space per capita :{round(pc_res,1)}
            - Commercial space per capit :{round(pc_comm,1)}
            - Office space per capita :{round(pc_office,1)}
            - Open space per capita :{round(pc_out,1)}   
            """
            )
        st.write(f"In this generated option, the number of residential occupants is {nafar_res1} with {Machine_res} cars,the number of office occupants is {nafar_office} with {Machine_office} cars, and number of commercial occupants is {nafar_comm} with {Machine_comm}. In this option the longest way is {longest_length} meter and shortest length is {lowest_length} meter. ")
        #st.write(f"The number of gasoline car is {num_gas_car} with {total_co2_gas} CO2 production")
        #if ad :
            #st.write(f"The number of CNG cars is {num_cng_car} with {total_co2_CNG} CO2 production.")
        #st.write(f"The total amount of CO2 production is {total_co2}")
        
        r_pv_building,e_c_building,e_h_building,e_l_building,en_h_building,hours_building,en_co_building,l_s_building,l_v_building=predictive_model(BuildingShape,green_ratio,SiteLength,var[0][0],var[1][0],var[2][0],Densityeachbldg,var[3][0],Lengths,Widths,u,v,zz,Area,adjacency,height,dimention,var[12][0],var[15][0],com_floor,res_com_loc,building_loc_copy)
        r_pv_park,e_c_park,e_h_park,e_l_park,en_h_park,hours_park,en_co_park,l_s_park,l_v_park=predictive_model_park(BuildingShape,green_ratio,SiteLength,var[0][0],var[1][0],var[2][0],Lengths,Widths,u,v,Area,height,dimention,park_loc)
        
             

        fig = go.Figure(layout={'scene': {'aspectmode':"data"}})
        fig.add_trace(go.Scatter3d(x=[Site_Width],y=[SiteLength],mode='markers',marker=dict(size=10,symbol='circle')))
        fig.update_layout(scene = dict(xaxis = dict(visible=True,dtick=20),yaxis = dict(visible=True,dtick=20),zaxis =dict(visible=True,backgroundcolor="whitesmoke")),annotations=[])
        x_center=[]
        y_center=[]
        for i in park_coor:
            x_center.append(i[0])
            y_center.append(i[1])
            
        for i,j in zip(x_center,y_center):
            fig.add_trace(park(12,0.1, i,j))
            
        for i,j,name in zip(x_center,y_center,range(1,len(y_center)+1)):
            name='P'+str(name)
            fig.add_trace(centerpark(0,0, i,j,name))#


        for x, y, z,l in zip(xx, yy, zz,res_com):
            fig.add_trace(towers([Lengths,Widths], z, x, y,l))
        for x, y, z,l in zip(xx2,yy2,zz2,commercial):
            fig.add_trace(comm([Lengths,Widths], z, x, y,l))
            
        count_res=0
        count_comm=0
        building_name=[]
        for i,j,z,name in zip(xx,yy,zz,res_com):        
            if name==0:
                legend='R'+str(count_res)
                building_name.append(legend)
                fig.add_trace(centerpark(0,z, i,j,legend))#
                count_res=count_res+1
            else:       
                legend='O'+str(count_comm)
                building_name.append(legend)
                fig.add_trace(centerpark(0,z, i,j,legend))#
                count_comm=count_comm+1
                         
 
        st.markdown("3D view")
        st.plotly_chart(fig)
        #st.write([len(i)for i in [zz,height,dimention,res_com_loc,building_loc_copy]])
        #st.write(res_com_loc,res_ratio)

        if BuildingShape==0:
            area=[(2*i*3.5*Widths)+(2*i*3.5*Lengths)+(Site_Width*SiteLength) for i in zz]
            vol=[area*sto*3.5 for area,sto in zip(area,zz)]
        else:
            area=[(2*i*3.5*Widths)+(2*i*3.5*Lengths)+(Site_Width*SiteLength)*75 for i in zz]
            vol=[area*sto*3.5*0.75 for area,sto in zip(area,zz)]

        #with svf,Location
        #each=pd.DataFrame({"building name":building_name,"location x":xx,"location y":yy,"Height":zz,"Aspect_ratio":[i/3.5 for i in zz],"surface_vol_ratio":[i/j for i,j in zip(area,vol)],'commerical_h':commercial,'PV generation (kWh/m2)':r_pv_building,"Cooling (kWh/m2)":e_c_building,"Heating (kWh/m2)":e_h_building,"Lighting (kWh/m2)":e_l_building,"Roof hot (kWh/m2)":en_h_building,"Solar Hours (Hours)":hours_building,"Roof Cold (kWh/m2)":en_co_building,"SVF %":l_s_building})

        #without svf,location
        
        flights=(sum(e_h_building)*0.21233 +sum(e_c_building)*0.18316 +sum(e_l_building)*0.21233)*Area+ total_co2
        f=(np.array([x + y for x, y in zip(e_c_building,e_l_building)]).dot(area))*0.18316+np.array(e_h_building).dot(area)*0.21233+ total_co2
        
        st.write(f"The total amount of CO2 production(Transportation and building emissions) is {round(f,2)} kg . This amount of carbon emissions is equivalent to {int(f/226)} one-way economy class flights for a single traveler between Tehran (IR) and Mashhad(IR) covering a distance of 800 km.")
   
        each=pd.DataFrame({"Building name":building_name,"Height":zz,"Number of Floors":[i/3.5 for i in zz],"Aspect ratio":[i/var[1][0]for i in zz],'commerical_h':commercial,
                           'PV generation (kWh/m2)':r_pv_building,"Cooling (kWh/m2)":e_c_building,"Heating (kWh/m2)":e_h_building,"Lighting (kWh/m2)":e_l_building,"Roof hot (kWh/m2)":en_h_building,
                           "Solar Hours (Hours)":hours_building,"Roof Cold (kWh/m2)":en_co_building })
        
        each['Co2']=((each['Cooling (kWh/m2)']*Area + each['Lighting (kWh/m2)']*Area)*0.21233+(each['Heating (kWh/m2)']*Area*0.18316))*stories
        #each['EUI(kWh/m2)']=((each['Cooling (kWh/m2)'] + each['Lighting (kWh/m2)'])*1.1+(each['Heating (kWh/m2)']*3.1))*0.9
        each['EUI(kWh/m2)'] = ((each['Cooling (kWh/m2)']*0.9 + each['Lighting (kWh/m2)'])*3.1 + (each['Heating (kWh/m2)']*0.9*1.1))


        
        # Convert columns to numeric, ignoring errors for non-convertible columns
        each[['PV generation (kWh/m2)', 'Cooling (kWh/m2)', 'Heating (kWh/m2)', 
              'Lighting (kWh/m2)', 'Roof hot (kWh/m2)', 'Solar Hours (Hours)', 
              'Roof Cold (kWh/m2)', 'Co2','EUI(kWh/m2)']] = each[['PV generation (kWh/m2)', 
              'Cooling (kWh/m2)', 'Heating (kWh/m2)', 'Lighting (kWh/m2)', 
              'Roof hot (kWh/m2)', 'Solar Hours (Hours)', 'Roof Cold (kWh/m2)', 
              'Co2','EUI(kWh/m2)']].apply(pd.to_numeric, errors='coerce')
        
        # Round numeric columns to 2 decimal places
        each = each.round({'PV generation (kWh/m2)': 2, "Cooling (kWh/m2)": 2, 
                           "Heating (kWh/m2)": 2, "Lighting (kWh/m2)": 2, 
                           "Roof hot (kWh/m2)": 2, "Solar Hours (Hours)": 2, 
                           "Roof Cold (kWh/m2)": 2, 'Co2': 2,'EUI(kWh/m2)':2})

        st.markdown("Outputs for each building")
        st.dataframe(each)
        #st.markdown("Outputs for generated neighbour")
        #st.dataframe(result)

        st.markdown("Outputs for Generated Parks")
        each_parks=pd.DataFrame({'name':['park '+str(i) for i in list(range(1,len(l_s_park)+1))],"location":park_coor,"SVF %":l_s_park,"Visibility %":l_v_park})
        each_parks[["SVF %","Visibility %"]].apply(pd.to_numeric, errors='coerce')
        
        # Round numeric columns to 2 decimal places
        each_parks = each_parks.round({"SVF %": 2,"Visibility %": 2})
        st.dataframe(each_parks)
        
        if Shade:
            shading1=shading(BuildingShape,var[0][0],building_coor,Lengths,Widths,SiteLength,Site_Width,zz)
            po=pd.DataFrame({"x":shading1[1],"y":shading1[2],"shading":shading1[0]})
            po['z']=1  
            fig, ax = plt.subplots()
            sc = ax.scatter(po['x'], po['y'], c=po['shading'], s=1, vmin=0.0, vmax=1.0)
            cbar = fig.colorbar(sc, ax=ax, label='Shading')


            isshaded=[]
            for i in shading1[0]:
                if i<0.90:
                    isshaded.append(1)
                else:
                    isshaded.append(0)
                    
            n_shade=round((sum(isshaded))/len(shading1[0]),1)   
            st.markdown(f"Shaded Area {n_shade} %")
            
            if round(n_shade*100,1) <= 20:
                st.write("""
                **Low Shading (0-20%)**
                
                **Winter:** 
                Maximizes solar heat gain, reducing heating demand.

                
                **Summer:** 
                Minimal shading increases cooling demand.

                
                **Additional Benefits:**
                PV production is at its highest, and natural daylight is fully utilized.

                
                **Thermal Comfort & Frost Risk:** 
                Outdoor spaces may become uncomfortably hot in summer, but the risk of frost or snow accumulation in winter is negligible.""")
                
            elif 20 < round(n_shade*100,1) <= 40:
                st.write("""
                **Moderate Shading (20-40%)**
                
                **Winter:** 
                Slightly reduces solar heat gain, leading to a minor increase in heating energy consumption.   
                
                
                **Summer:** 
                Provides moderate shading, improving cooling efficiency.
               
                
                **Additional Effects:** 
                PV production and daylight availability are moderately reduced. 
                
                
                **Thermal Comfort & Frost Risk:** 
                Offers improved thermal comfort in summer while maintaining a low risk of frost or snow accumulation in winter.
                """)
            elif 40 < round(n_shade*100,1) <= 60:
                st.write("""
                **High Shading (40-60%)**
                
                **Winter:** 
                Reduces solar heat gain significantly, resulting in higher heating demand.
                
                
                **Summer:** 
                Provides ample shading, significantly reducing cooling energy requirements.
                
                
                **Additional Effects:** 
                PV production and natural daylight are noticeably reduced.
                
                
                **Thermal Comfort & Frost Risk:** 
                Enhances thermal comfort in outdoor spaces during summer but increases the likelihood of frost or snow accumulation in shaded areas during winter.""")
            elif 60 < round(n_shade*100,1) <= 80:
                st.write("""
                **Very High Shading (60-80%)**
                
                **Winter:** 
                Significantly increases heating demand and sharply reduces PV production.
                
                
                **Summer:** 
                Creates excellent cooling conditions in outdoor spaces.
                
                
                **Additional Effects:** 
                Natural lighting is heavily diminished, increasing reliance on artificial lighting.
                
                
                **Thermal Comfort & Frost Risk:** 
                Provides optimal summer thermal comfort for outdoor spaces but heightens the risk of frost and snow buildup in winter, which may pose safety concerns. """)
            elif 80 < round(n_shade*100,1) <= 100:
                st.write("""
                **Maximum Shading (80-100%)**
                
                **Winter:** 
                Results in the highest heating demand while almost eliminating PV production.    
                
                
                **Summer:** 
                Maximizes cooling efficiency, ensuring the most comfortable outdoor conditions.
                
                
                **Additional Effects:** 
                Artificial lighting becomes essential due to minimal natural daylight, and the sky view is nearly completely obstructed.
                
                
                **Thermal Comfort & Frost Risk:** 
                Delivers the best thermal comfort in summer but carries the greatest risk of frost and snow accumulation in winter, potentially creating safety hazards in these areas.
                """)
        
            st.pyplot(fig)
            #st.set_option('deprecation.showPyplotGlobalUse', False)
            #fig = go.Figure(data=[go.Scatter3d(x=po['x'],y=po['y'],z=po['z'],mode='markers',marker=dict(size=0.0,color=po['shading']))])
            #fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        
                
        st.divider()
        
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        download.to_excel(writer, sheet_name='Sheet1')    
        # Close the Pandas Excel writer and output the Excel file to the buffer
        writer.close()
        st.download_button(label="Download optimization results ",data=buffer,file_name="pandas_multiple.xlsx",mime="application/vnd.ms-excel")

else:
    st.write("Tap Toggle to Optimize")
    
    
    

        
        



    
    
