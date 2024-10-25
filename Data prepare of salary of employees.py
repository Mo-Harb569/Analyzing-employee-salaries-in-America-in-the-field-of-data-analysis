# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:14:05 2024

@author: ASUS
"""
import numpy as np 
import pandas as pd
import category_encoders as ce
import seaborn as sns
import matplotlib.pyplot as plt 
from PIL import Image
import cv2 
from pytesseract import pytesseract
pytesseract.tesseract_cmd = (r"C:\Users\ASUS\Downloads\tesseract-ocr-w64-setup-5.3.3.20231005.exe")
print("Hello now we will analysis data set about data science")
data=pd.read_csv(r"C:\Users\ASUS\Downloads\archive (14)\DataScience_salaries_2024.csv")
print(data.columns)
data1 = pd.DataFrame(data)
print(data.tail(50))
print(data.head(50))
print(data.shape)
print(data.describe())
print(data.info())
print(data.describe(include ="int64"))
print(data.describe(exclude = "int64"))
print(data.describe(include = "object"))
print(data.describe(include = "all"))
subdata = data.loc[:10,["work_year","experience_level"]]
subdata2 = data.iloc[:10,3:5]
print(subdata)
print(subdata2)
subdata3 = data.iloc[10:50,2:8]
print(data.index)
type_=data.columns
print(type_)
print(data.dtypes)
print(data.select_dtypes("int64"))
print(data.select_dtypes(exclude = "object"))
print(data.select_dtypes(include = "object"))
print(data.values)
###############################
#علاقات 
work_years_count = data["work_year"].value_counts()
re_= data[(data["work_year"] >= 2020) & (data["work_year"] <= 2022)]
work_years_count20_22 = re_["work_year"].value_counts()
plt.pie(work_years_count20_22, labels=work_years_count20_22.index, autopct="%1.3f%%")
plt.title("Distribution of work years 20_22")
plt.show()

re__= data[(data["work_year"]>= 2022) & (data["work_year"]<=2024)]
work_years_count22_24 = re__["work_year"].value_counts()
plt.pie(work_years_count22_24,labels = work_years_count22_24.index,autopct = "%1.3f%%")
plt.title("Distribution of work years 20_24")
plt.show()

###########################################################################
#EN : مبتدأ , EX : نص خبير , MI : متوسط الخبره لكنه ليس خبير , SI : خبير 





# تحديد البيانات المطلوبة
isin_Low = data[data["experience_level"].isin(["EN","EX"])]
isin_High = data[data["experience_level"].isin(["MI","SE"])]
expereince_counts = data["experience_level"].value_counts()
most_common_ex = data["experience_level"].mode()[0]
print("The most common expereince is " , most_common_ex)
#هون عملت فلتره بين الاقل خبره والاعلى خبره 
###################################################
print(data["employment_type"].value_counts())
#{FL : دوام كامل ,CT : توظيف بالعقود , PT : توظيف بدوام جزئي , FT : توظيف حر }
FreeT_LowEX = data[(data["employment_type"]=="FT") & (data["experience_level"]=="EN")]
free_lowEX=data["employment_type"].value_counts()
plt.pie(free_lowEX,labels =free_lowEX.index,autopct ="%1.3f%%")
plt.show()
FreeT_HighEX = data[(data["employment_type"] =="FL") &(data["experience_level"] =="MI")]
free_highEX=data["employment_type"].value_counts()
plt.pie(free_highEX,labels =free_highEX.index,autopct ="%1.3f%%")
plt.show()

#هون عملت شويه فلتره للداتا بين ال دوام الكامل مع ادنى خبره والدوام الكامل والاقل خبره 
FreeLance_LowEX = FreeT_LowEX.shape[0]
FreeLance_HighEX = FreeT_HighEX.shape[0]

Total_FreeLance = FreeLance_LowEX + FreeLance_HighEX
percentage_FreeLance_LowEX = (FreeLance_LowEX / Total_FreeLance) * 100


percentage_FreeLance_HighEX = (FreeLance_HighEX / Total_FreeLance) * 100

print("The percentage of people work free and minimum expereince", percentage_FreeLance_LowEX, "%")
print("The percentage of people work free and medium expereince ", percentage_FreeLance_HighEX, "%")
#هيك انا طلعت النسبه المئويه للاشخاص الي بشتغلو حر مع خبره قليلقه و خبره متوسطه 
###################################################
#طيب هسا بدي اوجد بكل سنه كم شخص توظف في كل الاقسام تبع الدوام
 
year2020_EmTypeFL=(data["work_year"]==2020) & (data["employment_type"]=="FL")
year2020_EmTypeCT=(data["work_year"]==2020) & (data["employment_type"]=="CT")
year2020_EmTypePT=(data["work_year"]==2020) & (data["employment_type"]=="PT")
year2020_EmTypeFT=(data["work_year"]==2020) & (data["employment_type"]=="FT")
year2021_EmtypeFL=(data["work_year"]==2021) & (data["employment_type"]=="FL")
year2021_EmtypeCT=(data["work_year"]==2021) & (data["employment_type"]=="CT")
year2021_EmtypePT=(data["work_year"]==2021) & (data["employment_type"]=="PT")
year2021_EmtypeFL=(data["work_year"]==2021) & (data["employment_type"]=="FL")
year2022_EmtypeCT=(data["work_year"]==2022) & (data["employment_type"]=="CT")
year2022_EmtypePT=(data["work_year"]==2022) & (data["employment_type"]=="PT")
year2022_EmtypeFT=(data["work_year"]==2022) & (data["employment_type"]=="FT")
year2023_EmtypeFL=(data["work_year"]==2023) & (data["employment_type"]=="FL")
year2023_EmtypeCT=(data["work_year"]==2023) & (data["employment_type"]=="CT")
year2023_EmtypePT=(data["work_year"]==2023) & (data["employment_type"]=="PT")
year2023_EmtypeFT=(data["work_year"]==2023) & (data["employment_type"]=="FT")
year2024_EmtypeFL=(data["work_year"]==2024) & (data["employment_type"]=="FL")
year2024_EmtypeCT=(data["work_year"]==2024) & (data["employment_type"]=="CT")
year2024_EmtypePT=(data["work_year"]==2024) & (data["employment_type"]=="PT")
year2024_EmtypeFT=(data["work_year"]==2024) & (data["employment_type"]=="FT")
##################################################################################
data_len = len(data)
print(data_len) 

len_data=(data.isna().sum()/data_len) *100
print(len_data)
#هيك انا طلعت كم النسبه المئويه للداتا عشان ابلش اعالجها اكثر 
#الان اكتشفت انه ماعندي قيم missing بهاي الداتا بس رح اعمل كولوم جديد فيو نل واحذفه من باب العلم بالشيئ
NaN_col=data["new_NaN"]=np.nan
print(data.isna().sum())
#الان انا رح احذفه لانه ما بصير يكون عندل null 
#del data["new_NaN"]
p=data["new_NaN"].fillna(0,inplace = True)
print(data.isna().sum())
del data["new_NaN"]
print(data.info())

#######################################
#مرحله ال encoding 
x=data.select_dtypes(include = "object")
# object_columns =[Expereince_level ,employment type , job_title,salary_currency ,employee_residence ,company_location ,company_size   ]
print(data["experience_level"].value_counts())

#Now I will use the label encoding because the experience level is a value that can be arranged in the ordinal order.
data["experience_level"] = data["experience_level"].replace({"SE" : 1 , "MI" : 2 ,"EN" : 3 ,"EX" : 4})
print(data["employment_type"].value_counts())

#Now I will use the label encoding because the employment type  is a value that can be arranged in the ordinal order.
data["employment_type"] = data["employment_type"].replace({"FT" : 1 , "PT" : 2 , "CT" : 3 , "FL" : 4})


#Now I will use OneHot encoding because this coulom is a nominal value that cannot be arranged

data = pd.get_dummies(data, columns=['job_title'], drop_first=True)
#غدا سوال الدكتور عنه مهم عشان الكولومز الزائده الي طلعت معي 
#اتاكد من قيم ال unique

print(data["salary_currency"].value_counts())
#Now I will use the label encoding because the salary_currency is a value that can be arranged in the ordinal order.

# قم بترميز العمود salary_currency
encoded_salary_currency = pd.get_dummies(data['salary_currency'], prefix='salary_currency')
#delete the colum
# انضم الأعمدة المشفرة إلى البيانات الأصلية
data = pd.concat([data, encoded_salary_currency], axis=1)

# حذف العمود الأصلي
data.drop(['salary_currency'], axis=1, inplace=True)

print(data.info())
                     

#Now I will use the Binary encoding  because the employee_residence is a value that can be arranged in the ordinal order.
print(data["employee_residence"].value_counts())

unique_residence = data["employee_residence"].unique()
print(unique_residence)


encoded_employee_residence = pd.get_dummies(data['employee_residence'], prefix='employee_residence')
data = pd.concat([data, encoded_employee_residence], axis=1)

data.drop(['employee_residence'], axis=1, inplace=True)
print(data.info())


unique_residence = data["company_location"].unique()
print(unique_residence)

encoded_company_location = pd.get_dummies(data['company_location'], prefix='company_location')
data = pd.concat([data, encoded_company_location], axis=1)

data.drop(['company_location'], axis=1, inplace=True)
print(data.info())

#################################################
encoded_company_size = pd.get_dummies(data['company_size'], prefix='company_size')
data = pd.concat([data, encoded_company_size], axis=1)

data.drop(['company_size'], axis=1, inplace=True)
print(data.info())
#انتهت مرحله ال encoding 

###################################################
print(data.columns)
int_columns = data.select_dtypes(include=["int64"]).columns
print(int_columns)
#scalling 
#تم استثناء الكولوم salary لانه يعتبر target وقد يؤثر بالتنبؤ لاحقا سلبا 
from sklearn.preprocessing import StandardScaler

columns_to_scale = ["work_year",  "salary_in_usd", "remote_ratio"]

scaler = StandardScaler()

data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

des2 = data.describe().round(2).T
print(des2)

print(data.columns)

#Groupby
avg_salary_by_year = data.groupby("work_year")["salary_in_usd"].mean()
print(avg_salary_by_year)


total_salary_by_job_title = data.groupby("job_title_AI Developer")["salary_in_usd"].sum()
print(total_salary_by_job_title)


job_count_by_company_location = data.groupby("company_location_IR").size()
print(job_count_by_company_location)


salary_by_year_and_job_title = data.groupby(["work_year", "job_title_Lead Data Scientist"])["salary_in_usd"].mean()
print(salary_by_year_and_job_title)


salary_range_by_experience_level = data.groupby("experience_level")["salary_in_usd"].agg(["min", "max"])
print(salary_range_by_experience_level)

print(data.columns)
print(data.head())

plt.figure(figsize=(16,8))

# قيم النسب المئوية
percentile25 = data["salary"].quantile(0.25)
percentile75 = data["salary"].quantile(0.75)
iqr = percentile75 - percentile25
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr

# تصفية البيانات لإزالة القيم المتطرفة
new_data = data[(data["salary"] >= lower_limit) & (data["salary"] <= upper_limit)]

# رسم الرسوم البيانية
plt.subplot(2,2,1)
sns.distplot(data["salary"])
plt.title('Original Data')

plt.subplot(2,2,2)
sns.boxplot(data["salary"])
plt.title('Original Data')

plt.subplot(2,2,3)
sns.distplot(new_data["salary"])
plt.title('Data without Outliers')


plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 4)
sns.boxplot(data=new_data, x="salary")
plt.title('Data without Outliers', fontsize=14)
plt.xlabel('Salary', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()
#pandas profilling
#code 892
#corr = data.corr()
#print(corr)
#خليه للدكتور بكره بساله كيف بدي احذف
img = Image.open(r"C:\Users\ASUS\Desktop\Image Resizing for Blog and Social (1)-Nov-23-2020-09-19-42-03-PM.webp")
#open image using pillow
imag_cv = cv2.imread(r"C:\Users\ASUS\Desktop\Image Resizing for Blog and Social (1)-Nov-23-2020-09-19-42-03-PM.webp")
#open image using open_cv

print(type(img))
print(type(imag_cv))
#Show this image 
cv2.imshow("Data sceintest employee",imag_cv)
cv2.waitKey(1000)
cv2.destroyAllWindows()
#Rotate an image 
h = imag_cv.shape[0]
w = imag_cv.shape[1]
rotation_matrix=cv2.getRotationMatrix2D((w/2,h/2),45,.5)
rotated_image =cv2.warpAffine(imag_cv,rotation_matrix,(w,h))
cv2.imshow("Data sceintest employee rotated",rotated_image)
cv2.waitKey(5000)
cv2.destroyAllWindows()
# Creat crop with image 
startRow = int(h*.15)
startCol = int(w*.15)
endRow = int(h*.85)
endCol = int(w*.85)


croppedImage = imag_cv[startRow:endRow,startCol:endCol]
cv2.imshow("original Image" , imag_cv)
cv2.imshow("cropped Image",croppedImage)
cv2.waitKey(5000)
cv2.destroyAllWindows()

# Resizing an image 
Resize_Imag = cv2.resize(imag_cv,(0,0),fx =.5,fy = .5)
cv2.imshow("original Image" , imag_cv)
cv2.imshow("Resize_Imag" , Resize_Imag)
cv2.waitKey(5000)
cv2.destroyAllWindows()
# make image blury 
#gussein blury 
blur_Image = cv2.GaussianBlur(imag_cv,(17,17),0)
cv2.imshow("original Image" , imag_cv)
cv2.imshow("blur_Image" , blur_Image)
cv2.waitKey(5000)
cv2.destroyAllWindows()
#convert image to grayscale image 
imag_cv = cv2.imread(r"C:\Users\ASUS\Desktop\Image Resizing for Blog and Social (1)-Nov-23-2020-09-19-42-03-PM.webp",0)
cv2.imshow("original Image" , imag_cv)
cv2.waitKey(1000)
cv2.destroyAllWindows()
###############################
#extract from image to string 
image = Image.open(r"C:\Users\ASUS\Desktop\maxresdefault.jpg")
extr = pytesseract.image_to_string(image,"eng")
print(extr)
