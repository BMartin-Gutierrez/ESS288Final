##Brian Martin##
##ES 288 Final##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


file_path = r"C:\Users\Brian\Desktop\ESS288Final-main\Final Project\VehiclesClean3.csv"
linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),
     ('dashdotted',            (0, (3, 5, 1, 5)))]

DesiredColumns = ["city08", "co2TailpipeGpm", "comb08", "cylinders", "drive", "fuelType1", "highway08", "make", "model",
                  "VClass", "year"]
makes = ['BMW', 'Chevrolet', 'Chrysler', 'Ford', 'Honda',
         'Mazda', 'Mercedes-Benz', 'Nissan', 'Toyota', 'Volkswagen', 'Volvo', 'Hyundai', 'Kia']
fuelType = ["Regular Gasoline", "Midgrade Gasoline", "Premium Gasoline"]

# Clean the Data
df = pd.read_csv(file_path, usecols=DesiredColumns, engine='python')
df.drop_duplicates(subset=['year', 'model'], keep="first", inplace=True)
df_reduced_makes = df.loc[df['make'].isin(makes)]
df_reduced = df_reduced_makes.loc[df_reduced_makes['fuelType1'].isin(fuelType)]
df_reduced = df_reduced[df_reduced['model'].str.contains('Wagon|Convertible|Cabriolet|Coupe|Touring|Hybrid|Natural '
                                                         'Gas|Turbo|e-tron|quattro|sport|amg|2wd|diesel|srt|XLE|XSE|AWD',
                                                         case=False, na=False) == False]

data = []
FirstYear = df_reduced["year"].min()+2
LatestYear = df_reduced["year"].max()-3
Years = LatestYear - FirstYear

#Get Avg Mpg data for each manufacturer
for make in makes:
    MakeData = df_reduced.loc[df_reduced["make"] == make]

    YearRange = range(FirstYear, LatestYear)

    for year in YearRange:
        MakeYearData = MakeData.loc[df_reduced["year"] == year]
        MakeYearCars = MakeYearData.loc[MakeData["VClass"].str.contains("car", case=False, na=False)]
        MakeYearTrucks = MakeYearData.loc[MakeData["VClass"].str.contains("truck", case=False, na=False)]

        AvgMpgYearCar = MakeYearCars["comb08"].mean()
        AvgMpgYearTruck = MakeYearTrucks['comb08'].mean()
        data.append([AvgMpgYearCar, AvgMpgYearTruck, make, year])

## Create df with avg mpg data
df_AvgMpg = pd.DataFrame(data, columns=['CarAvgMpg', 'TruckAvgMpg', 'Make', 'Year'])
df_AvgMpg.dropna(subset=["CarAvgMpg"],inplace=True)
i=0

#Plot
for make in makes:
    DataSubset = df_AvgMpg.loc[df_AvgMpg["Make"] == make]

    ## Quadratic Regression for points
    CarModel = np.poly1d(np.polyfit(DataSubset["Year"], DataSubset["CarAvgMpg"], 1))
    # TruckModel = np.poly1d(np.polyfit(DataSubset["Year"], DataSubset["TruckAvgMpg"], 2))
    polyline = np.linspace(FirstYear, LatestYear, Years, endpoint=True)
    #
    print(len(DataSubset["CarAvgMpg"]),len(polyline))
    if len(DataSubset["CarAvgMpg"]) == len(polyline):
        CarR2 = r2_score(DataSubset["CarAvgMpg"], CarModel(polyline))
        print(make, CarR2)
    else:
        continue

    plt.plot(polyline, CarModel(polyline), label=make, linestyle= linestyle_tuple[i][1])
    i+=1
    # plt.plot(polyline, TruckModel(polyline),label = "FordTruck")


# df_AvgMpg.to_csv(r"C:\Users\Brian\Desktop\ESS288Final-main\Final Project\VehiclesClean5.csv")
# labelLines(plt.gca().get_lines(), zorder=2.5)
# plt.savefig(r"C:\Users\Brian\Desktop\ESS288Final-main\Final Project\ToyotaHex.pdf")
# plt.legend()
plt.show()
