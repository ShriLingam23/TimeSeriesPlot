from TimeSeriesData import TimeSeriesData
import matplotlib.pyplot as plt

ts_data = TimeSeriesData(250,0,10)
plt.plot(ts_data.x_data,ts_data.y_true)
plt.show()
