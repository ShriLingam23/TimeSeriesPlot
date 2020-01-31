from TimeSeriesData import TimeSeriesData
import matplotlib.pyplot as plt

#Just plot the graph for the given data
ts_data = TimeSeriesData(250,0,10)
#plt.plot(ts_data.x_data,ts_data.y_true)
#plt.show()


#Predict one point ahead in the time series
num_time_steps=30

y1,y2,ts= ts_data.next_batch(1,num_time_steps,True)

print(ts.flatten().shape)

#plt.plot(ts.flatten()[1:],y1.flatten(),"*")
#plt.show()

plt.plot(ts_data.x_data,ts_data.y_true)
plt.plot(ts.flatten()[1:],y1.flatten(),"g*")
plt.show()