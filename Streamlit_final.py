#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import streamlit as st
from scipy import stats
from PIL import Image

# Data Process

# process cost data
class cost_process():
    def __init__(self, data, state, area):
        # data['GROCERY_ITEMS'] = data['GROCERY_ITEMS'].astype(float)
        self.data_2021 = data
        self.state = state
        self.area = area

    
    def avg_state(self):
        cost_2021 = self.data_2021[['COMPOSITE_INDEX', 'URBAN_AREA_NAME', 'STATE_NAME']]
        # remove quater
        cost_avg_area = cost_2021.groupby(['URBAN_AREA_NAME','STATE_NAME']).agg(np.mean)
        # state avg / var cost
        cost_avg_stat = cost_avg_area.groupby(['STATE_NAME']).agg(np.mean).sort_values('COMPOSITE_INDEX')
        cost_var_stat = cost_avg_area.groupby(['STATE_NAME']).agg(np.var).sort_values('COMPOSITE_INDEX')
        return cost_avg_stat, cost_var_stat, cost_avg_area

    def rank(self):
        cost_avg_stat, cost_var_stat, cost_avg_area = self.avg_state() # get dataset needed
        # state = cost_avg_area.loc[area,'COMPOSITE_INDEX'].index[0] # get state
        state_cost = cost_avg_stat.loc[self.state, 'COMPOSITE_INDEX'] # get cost
        avg_rank = cost_avg_stat.index.get_loc(self.state) + 1
        var_rank = cost_var_stat.index.get_loc(self.state) + 1
        
        return state_cost, avg_rank, var_rank
    
    def clust(self):
        data_2021_clu = self.data_2021[['GROCERY_ITEMS', 'HOUSING', 'TRANSPORTATION', 'UTILITIES', 'HEALTH_CARE', 'MISC_GOODS_SERVICES',
                           'URBAN_AREA_NAME', 'STATE_NAME']]
        data_2021_clu['GROCERY_ITEMS'] = data_2021_clu['GROCERY_ITEMS'].astype(float)
        clu_avg_area = data_2021_clu.groupby(['URBAN_AREA_NAME','STATE_NAME']).agg(np.mean)
        # kmeans
        kmeans = KMeans(n_clusters=4, random_state=0).fit(clu_avg_area.values.tolist())
        clu_avg_area['clusters']=kmeans.labels_
        
        return clu_avg_area
    
    def clust_desc(self):
        # state = self.state # get state
        clu_avg_area = self.clust() # get dataset
        clu_label = ['LOW', 'MID', 'HIGH', 'VERY HIGH'] # set cluster description
        cost_avg_area = self.data_2021[['COMPOSITE_INDEX', 'URBAN_AREA_NAME', 'STATE_NAME']].groupby(['URBAN_AREA_NAME','STATE_NAME']).agg(np.mean)
        area_cost = list(cost_avg_area.loc[self.area,self.state])[0] # get area cost
        area_cluster = clu_avg_area.loc[self.area,self.state][6] # get cluster
        area_description = clu_label[int(area_cluster)]
        
        return area_cost, area_description

class health_process():
    def __init__(self, data): # input data
        self.data = data
    
    def percentile(self, county):
        data = self.data 
        user_level = list(data[data["County State"] == county]["weighted sum"])[0]
        percentile = round(stats.percentileofscore(data["weighted sum"].values, user_level), 2)
        
        return percentile

# income data
class income_process():
    def __init__(self, filename, UserIncome): # input data
        self.RegionData = pd.read_excel(filename, sheet_name = 0)
        self.HouseholdData = pd.read_excel(filename, sheet_name = 1)
        self.RaceData = pd.read_excel(filename, sheet_name = 2)
        self.EducationData = pd.read_excel(filename, sheet_name = 3)
        self.AgeData = pd.read_excel(filename, sheet_name = 4)
        self.GenderData =  pd.read_excel(filename, sheet_name = 5)
        
        self.judge = ['below', 'above']
        self.UserIncome = UserIncome
    
    def region(self, state):
        RegionData = self.RegionData
        for i in RegionData['Region']:
            if state == i :
                rowindex = RegionData.index[RegionData['Region'] == i].to_list()
                LocalIncMedian = RegionData['2021RegionIncome'][rowindex[0]]
                break
        
        region_star = int(self.UserIncome > LocalIncMedian)
        region_judge = self.judge[region_star]
        
        return region_star, region_judge, LocalIncMedian
    
    def house(self, Household):
        HouseholdData = self.HouseholdData
        for h in HouseholdData['Household']:
            if Household == h :
                rowindex = HouseholdData.index[HouseholdData['Household'] == h].to_list()
                HouseholdIncMedian = HouseholdData['2021HouseholdIncome'][rowindex[0]]
                break
                
        house_star = int(self.UserIncome > HouseholdIncMedian)
        house_judge = self.judge[house_star]
        
        return house_star, house_judge, HouseholdIncMedian
    
    def race(self, Race):
        RaceData = self.RaceData
        for i in RaceData['Race']:
            if Race == i :
                rowindex = RaceData.index[RaceData['Race'] == i].to_list()
                RaceIncMedian = RaceData['2021RaceIncome'][rowindex[0]]
                break
                
        race_star = int(self.UserIncome > RaceIncMedian)
        race_judge = self.judge[race_star]
        
        return race_star, race_judge, RaceIncMedian
        
    def age(self, Age):
        AgeData = self.AgeData
        for i in AgeData['Age']:
            if Age == i :
                rowindex = AgeData.index[AgeData['Age'] == i].to_list()
                AgeIncMedian = AgeData['2021AgeIncome'][rowindex[0]]
                break
        
        age_star = int(self.UserIncome > AgeIncMedian)
        age_judge = self.judge[age_star]
        
        return age_star, age_judge, AgeIncMedian
    
    def edu(self, Education):
        EducationData = self.EducationData
        for i in EducationData['Education']:
            if Education == i :
                rowindex = EducationData.index[EducationData['Education'] == i].to_list()
                EducationIncMedian = EducationData['2021EducationIncome'][rowindex[0]]
                break
        
        edu_star = int(self.UserIncome > EducationIncMedian)
        edu_judge = self.judge[edu_star]
        
        return edu_star, edu_judge, EducationIncMedian
    
    def gender(self, Gender):
        GenderData = self.GenderData
        for i in GenderData['Gender']:
            if Gender == i :
                rowindex = GenderData.index[GenderData['Gender'] == i].to_list()
                GenderIncMedian = GenderData['2021GenderIncome'][rowindex[0]]
                break
        
        gen_star = int(self.UserIncome > GenderIncMedian)
        gen_judge = self.judge[gen_star]
        
        return gen_star, gen_judge, GenderIncMedian
    
    def process(self, state, Household, Race, Age, Education, Gender):
        region_star, region_judge, LocalIncMedian = self.region(state)
        house_star, house_judge, HouseholdIncMedian = self.house(Household)
        race_star, race_judge, RaceIncMedian = self.race(Race)
        age_star, age_judge, AgeIncMedian = self.age(Age)
        edu_star, edu_judge, EducationIncMedian = self.edu(Education)
        gen_star, gen_judge, GenderIncMedian = self.gender(Gender)
        
        income_outcome = {"User's Infomation": [state, Household, Race, Age, Education, Gender],
                        'Benchmark': [LocalIncMedian, HouseholdIncMedian, RaceIncMedian, AgeIncMedian, EducationIncMedian, GenderIncMedian],
                        "Whether Get Star ⭐": ['⭐'*region_star, '⭐'*house_star, '⭐'*race_star, '⭐'*age_star,  '⭐'*edu_star, '⭐'*gen_star]}
        income_outcome = pd.DataFrame(income_outcome)
        income_outcome.index = ['State', 'Gender', 'Age', 'Race', 'Household Type', 'Education Level']
        
        if Age == '15 to 24 years':
            edu_count = 0
        else:
            edu_count  =1
        
        Star = region_star+house_star+race_star+age_star+edu_star*edu_count+gen_star
        
        if Star in range(0,3):
            income_status = 'LOW'
        elif Star in range(3,5):
            income_status = 'MEDIAN'
        else:
             income_status = 'HIGH'
        
        # benchmark
        RegionData = self.RegionData
        rowindex = RegionData.index[RegionData['Region'] == 'United States'].to_list()
        USIncMedian = RegionData['2021RegionIncome'][rowindex[0]]
        
        return region_judge, house_judge, race_judge, age_judge, edu_judge, gen_judge, income_status, USIncMedian, Star, income_outcome

class ifinput():
    def __init__(self, cost_data, health_data, income_filename):
        self.cost_data = cost_data
        self.health_data = health_data
        self.income_filename = income_filename
    
    def run(self, state, area, county, UserIncome, Household, Race, Age, Education, Gender):
        # cost
        if area in list(self.cost_data['URBAN_AREA_NAME']): # if the area is not in database
            cost = cost_process(self.cost_data, state, area)
            state_cost, avg_rank, var_rank = cost.rank()
            area_cost, area_description = cost.clust_desc()
        else:
            area_cost, area_description = [-1, -1]
            state_cost, avg_rank, var_rank = [-1, -1, -1]
        
        # health
        if county in list(self.health_data['County State']):
            health = health_process(self.health_data)
            health_percentile = health.percentile(county)
        else:
            health_percentile = -1
        
        # income
        income = income_process(self.income_filename, UserIncome)
        region_judge, house_judge, race_judge, age_judge, edu_judge, gen_judge, income_status, USIncMedian, Star, income_outcome = income.process(state, Household, Race, Age, Education, Gender)
        # output dictionary
        key = ['state_cost', 'avg_rank', 'var_rank','area_cost', 'area_description',
               'health_percentile',
               'region_judge', 'house_judge', 'race_judge', 'age_judge', 'edu_judge', 'gen_judge', 'income_status', 'USIncMedian', 'Star']
        value = [state_cost, avg_rank, var_rank,area_cost, area_description, 
                 health_percentile,
                 region_judge, house_judge, race_judge, age_judge, edu_judge, gen_judge, income_status, USIncMedian, Star]
        output = dict(zip(key, value))
        
        return output, income_outcome

# Interface

# import data
data_cost = pd.read_excel('cost.xlsx')
data_health = pd.read_excel("combined data.xlsx")
income_file = "incomeComb.xlsx"

# general information
st.title('🔍 Sayhii Interface')
st.markdown('''##### <span style="color:orange">Analyse and present benchmarks of 3 external factors impacting the wellness level of user according to user's properties</span>
            ''', unsafe_allow_html=True)

# input information
UserIncome =  st.number_input('💵 Enter Your Income Per Year ($): ')

state = st.selectbox('Choose State: ',('Mississippi', 'West Virginia','New Mexico','Louisiana','Arkansas',
'Alabama','Kentucky','Oklahoma','Florida','Tennessee','North Carolina','Georgia','South Carolina','Montana','Missouri'
,'Michigan','Ohio','Maine','Nevada','Indiana','Texas','Wyoming','South Dakota','Idaho','North Dakota','Arizona','Wisconsin'
,'Iowa','Pennsylvania','New York','Delaware','Kansas','Nebraska','Vermont','Rhode Island','Illinois','Oregon','California'
,'Alaska','Minnesota','Colorado','Virginia','Connecticut','Washington','Utah','Hawaii','New Jersey','New Hampshire','Massachusetts'
,'District of Columbia' ,'Maryland') )

area_select = list(set(list(data_cost[data_cost['STATE_NAME'] == state]['URBAN_AREA_NAME'])))
area_select.append('Unknown')
area = st.selectbox('Choose Area: ', area_select)

county_select = list(data_health[data_health['State'] == state]['County State'])
county_select.append('Unknown')
county =  st.selectbox('Choose County: ', county_select)

Gender = st.selectbox(
    'Choose Gender:',
    ('Male','Female'))


Age = st.selectbox(
    'Choose Age Range:',
    ('15 to 24 years','25 to 34 years','35 to 44 years','45 to 54 years','55 to 64 years','65 years and older'))


Race = st.selectbox(
    'Choose Race:',
    ('White','Black','Asian','Hispanic (any race)'))


family = st.selectbox(
    'Family Households?',
    ('Yes', 'No'))

if family == 'Yes':
    Household = st.selectbox(
        'If Family Households, Please Choose Household Type Here:',
        ('Married-couple', 'Female householder, no spouse present', 'Male householder, no spouse present'))
    
else:
    Household = st.selectbox(
        'If Non-family Households, Please Choose Household Type Here:',
        ('Female householder','Male householder'))
   
# edu
Education = st.selectbox(
    'Choose Education Level:',
    ('No high school diploma', 'High school, no college', 'Some college',"Bachelor's degree or higher"))

    

    
    

if st.button('Continue'):
    
    # get input
    input_data = ifinput(data_cost, data_health, income_file)
    output_dict, income_outcome = input_data.run(state, area, county, UserIncome, Household, Race, Age, Education, Gender)
    
    # get output
    # general
    st.write('\n*********************************************************************************\n')
    st.write('This person lives in '+ state)
    if Gender == 'Male':
        p = 'His'
    else:
        p = 'Her'
    st.write(p + ' annual income is $' + str(int(UserIncome)) + '.')
    
    tab1, tab2, tab3 = st.tabs(['Cost of living', 'Healthcare', 'Income'])

    # cost
    with tab1:
        st.markdown('''#### User's Cost Level:''', unsafe_allow_html=True)
        if area in list(data_cost['URBAN_AREA_NAME']): 
            st.write('The state ranks ' + str(output_dict['avg_rank']) + ' in low cost of living' )
            st.write('The state ranks ' + str(output_dict['var_rank']) + ' in small gap of in-state cost of living')
            # st.write('The average of cost in '+ area + ' is '+ str(output_dict['area_cost']))
            st.write('This area is '+ output_dict['area_description'] + ' cost area')

            cost_description = {'GROCERY_ITEMS': ['low','median','median','high'],
                               'HOUSING': ['low','median','high','very high'],
                               'TRANSPORTATION': ['low','median','high','high'],
                               'UTILITIES': ['low','median','high','median'],
                               'HEALTH_CARE': ['low','median','median','median'],
                               'MISC_GOODS_SERVICES': ['low','median','high','high']}
            cost_table = pd.DataFrame(cost_description)
            cost_table.index = ['low','median','high','very high']
            st.table(cost_table)

        else: 
            st.write('This area is not in the cost database')
            
    # health
    with tab2:
        st.markdown('''#### User's Health Level:''', unsafe_allow_html=True)
        if county in list(data_health['County State']):
            st.write("The level of access to healthcare level at the user's residency is better than " 
                     + str(output_dict['health_percentile']) + "% of all locations in the U.S.")
        else: 
            st.write('This county is not in the health database')
        health_img = Image.open('health.png')
        st.image(health_img)

    # income
    with tab3:
        st.markdown('''#### User's Income Level:''', unsafe_allow_html=True)

        st.markdown('''##### <span style="color:gray"> **Notes:** Benchmark is the median income in 2021 for the people with related property. If user's income is above the benchmark, get 1 star ⭐ </span>
            ''', unsafe_allow_html=True)
        
        st.markdown('''##### <span style="color:orange">Explanation For Above Income Level Table</span>
            ''', unsafe_allow_html=True)
        st.write("This person's income is " + output_dict['region_judge'] + " the median level of " + state)
        st.write("This person's income is " + output_dict['gen_judge'] + " the median level of " + Gender)
        st.write("This person's income is " + output_dict['age_judge'] + " the median level of " + Age + ' people')
        st.write("This person's income is " + output_dict['race_judge'] + " the median level of " + Race)
        st.write("This person's income is " + output_dict['house_judge'] + " the median level of " + Household)
        if Age == '15 to 24 years':
            st.write("As you are under 25, we will not judge the income based on your education.")
            income_outcome_underage = income_outcome.drop(index = 'Education Level')
            st.table(income_outcome_underage)
            Star_total = 5
        else:
            st.write("This person's income is " + output_dict['edu_judge'] + "the median level of people whose highest education level is " + Education)
            st.table(income_outcome)
            Star_total = 6
        
        st.markdown('''##### <span style="color:orange">Total Stars And Income Level For This User</span>
            ''', unsafe_allow_html=True)
        st.write('*Income Status:* '+ '⭐' * output_dict['Star'] + '🟨'* (Star_total - output_dict['Star']) )
        st.write('*Income Level:* ' + output_dict['income_status'])

        

            


# In[ ]:


# sidebar

image = Image.open('sayhii.png')

st.sidebar.image(image)

st.sidebar.markdown(" ## About Sayhii Interface")
st.sidebar.markdown("This presentation model shows us the user's cost, health and income level according to his/her properties, like location, gender, age, education level etc. Then Sayhii can understand how these three external factors for the wellness level impact its users."  )              
st.sidebar.info("Read more about how this model works and see the code on the [Github](https://github.com/danshi0109/Sayhii-Streamlit).")

