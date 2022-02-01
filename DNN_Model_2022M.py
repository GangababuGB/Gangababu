import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# print(tf.__version__)

# def plot_series(time, series, format="-", start=0, end=None):
#     plt.plot(time[start:end], series[start:end], format)
#     plt.xlabel("Time")
#     plt.ylabel("Value")
#     plt.grid(True)
#     plt.show()

time_step = []# index
qty = []
with open('Mean_WEEK-WISE_FGAS-660A-16-00087_BC - Copy.csv') as csvfile:
  reader = csv.reader(csvfile, delimiter=',')
  next(reader)
  for i in reader:
      qty.append(int(i[1])) # CLIP custom linear IP Datta
      time_step.append((i[0]))

series = np.array(qty)
time = np.array(time_step) # index
# plt.figure(figsize=(10, 6))
# plot_series(time, series)
# plt.plot(time, series)
# plt.show()

split_time = int(1 * len(series))
print(len(series))
print(split_time)
print(len(series)-split_time)
time_train = time[:split_time]# index
x_train = series[:split_time] # QTY
time_valid = time[split_time:]# index
x_valid = series[split_time:] # QTY

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset

tf.keras.backend.clear_session()

window_size = 2
shuffle_buffer_size = len(series-2)
batch_size = shuffle_buffer_size-2

lr=0.001 # default 0.001, rho=0.9, epsilon=1e-08, decay=0.0
rho=0.9
# epsilon=1e-08
# decay=0.0
# batch_size = 50
# shuffle_buffer_size = 51
# dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
dataset = windowed_dataset(series, window_size, batch_size, shuffle_buffer_size)

# relu | LeakyReLU | gelu | elu | selu | swish
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(20, input_shape=[window_size], activation="LeakyReLU"),
#     tf.keras.layers.Dense(10, activation="LeakyReLU"),
#     tf.keras.layers.Dense(1, activation="LeakyReLU")
# ])

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(8, input_shape=[window_size], activation="LeakyReLU"),
    tf.keras.layers.Dense(4, activation="LeakyReLU"),
    tf.keras.layers.Dense(1, activation="LeakyReLU")
])

print('Model Summary \n', model.summary())

# LR __________________________________________________________________
# lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
# model.compile(loss="mae", optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9))
# history = model.fit(dataset, epochs=100, callbacks=[lr_schedule])
# plt.semilogx(history.history["lr"], history.history["loss"])
# mng = plt.get_current_fig_manager()
# mng.window.state("zoomed")
# # plt.axis([1e-8, 1e-4, 0, 800])
# plt.show()

# LR __________________________________________________________________
# lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
# model.compile(loss="mae", optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0))
# history = model.fit(dataset, epochs=120, callbacks=[lr_schedule])
# plt.semilogx(history.history["lr"], history.history["loss"])
# mng = plt.get_current_fig_manager()
# mng.window.state("zoomed")
# # plt.axis([1e-8, 1e-4, 0, 800])
# plt.show()

# LR __________________________________________________________________
# lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
# opt=tf.keras.optimizers.Ftrl(learning_rate=0.001, learning_rate_power=-0.5, initial_accumulator_value=0.1,
#                              l1_regularization_strength=0.0, l2_regularization_strength=0.0,
#                              name='Ftrl', l2_shrinkage_regularization_strength=0.0, beta=0.0)
# model.compile(loss="mae", optimizer=opt)
# history = model.fit(dataset, epochs=100, callbacks=[lr_schedule])
# plt.semilogx(history.history["lr"], history.history["loss"])
# mng = plt.get_current_fig_manager()
# mng.window.state("zoomed")
# # plt.axis([1e-8, 1e-4, 0, 800])
# plt.show()
# # _______________________________________________________________________

# # _______________________________________________________________________

# # from keras.optimizers import Adagrad, Adadelta, RMSprop, Adam
# lr = 5.54e-05
epochs=2000
# opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
# opt = tf.keras.optimizers.Adagrad(learning_rate=0.01, epsilon=1e-08, decay=0.0)
# lr = 1 # default 1.0
# rho=0.95 # default 0.95 epsilon=1e-08
# opt = tf.keras.optimizers.Adadelta(learning_rate=lr, rho=0.95, epsilon=1e-08, decay=0.0)
# lr=0.0011 # default 0.001, rho=0.9, epsilon=1e-08, decay=0.0
# rho=0.94
# rho=0.95
# opt = tf.keras.optimizers.RMSprop(learning_rate=lr, rho=rho, epsilon=epsilon, decay=decay)
# rho=0
# lr=0
# opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
opt=tf.keras.optimizers.Ftrl(learning_rate=0.005, learning_rate_power=-0.5, initial_accumulator_value=0.1,
                             l1_regularization_strength=0.0, l2_regularization_strength=0.0,
                             name='Ftrl', l2_shrinkage_regularization_strength=0.0, beta=0.0)
# opt=tf.keras.optimizers.Nadam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam')
# opt=tf.keras.optimizers.Adamax(learning_rate=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Adamax')

# //////////////////////////////////////////////////////
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
model.compile(loss="mae", optimizer=opt)
history = model.fit(dataset, epochs=epochs, callbacks=[callback], verbose=1)

print('len of epoch: ', len(history.history['loss']))
# print('dataset shape:\n', dataset.shape)

# forecast = []
# for i in range(len(series) - window_size):
#   forecast.append(model.predict(series[i:i + window_size][np.newaxis]))

epochs= len(history.history['loss'])

# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# predicted = model.predict(X_train)
# series = np.reshape(series, (series.shape[0], series.shape[1], 1))
# predicted = model.predict(series)
# print('predicted:\n', predicted)

forecast = []
for i in range(len(series) - window_size):
    forecast.append(model.predict(series[i:i + window_size][np.newaxis]))

print('forecast:\n', forecast)
# forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]
# results = np.roll(results, 1)
print('results:\n', results)
print('LO_results: ', len(results)) #

x_train=x_train[:len(results)]
print(len(x_train)) #
time_train=time_train[:len(results)]
print(len(time_train))

MAE = tf.keras.metrics.mean_absolute_error(x_train, results).numpy()
MAPE = round(100-tf.keras.metrics.mean_absolute_percentage_error(x_train, results).numpy(), 2)
print('MAE: ', MAE)
print('Accuracy: ', 100-MAPE)
plt.figure(figsize=(10, 6))
plt.plot(time_train, x_train, label='Test_QTY')
plt.plot(time_train, results, label='Pred_QTY')
plt.title('KPL time series prediction - Actual Vs. Predicted [ ACC:'+str(MAPE)+
          " ] Epochs: " +str(epochs)+ ' rho: ' +str(rho)+' [ LR:' + str(lr)+" ]")

plt.xlabel("Time")
plt.ylabel("QTY")
# for x, y in zip(time_valid, x_valid):
#     label = int(y)
#     plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 1), ha='center')
# for x, y in zip(time_valid, results):
#     label = int(y)
#     plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 1), ha='center')
plt.xticks(rotation=90, size=8)
plt.subplots_adjust(bottom=0.2, left=0.06, right=0.96, top=0.9)
plt.grid(color='grey', linestyle='dotted', linewidth=0.5)
plt.grid(True, color='grey', which='major', alpha=0.5)
plt.grid(True, color='grey', which='minor', alpha=0.5)
mng = plt.get_current_fig_manager()
mng.window.state("zoomed")
plt.legend()
plt.show()

time_train = pd.DataFrame(time_train, columns=['DATE'])
print('time_train_H:\n', time_train.head())
print('time_train_T:\n', time_train.tail())
print('len^:', len(time_train))

x_train = pd.DataFrame(x_train, columns=['QTY'])
print('x_train:\n', x_train.tail())
print('len^:', len(x_train))

PRED = pd.DataFrame(results, columns=['PRED'])
PRED.PRED = PRED.PRED.astype(int)
print('time_train:\n', PRED.tail())
print('len^:', len(PRED))

C_df = pd.concat([time_train, x_train, PRED], axis=1)
print(C_df.tail())

# #-----------------------------------------------------------
# forecast for next 4 weeks
n_features = window_size
# results = results.astype(int)
x_input = np.array(results)[len(results)-window_size:]
temp_input = list(x_input)
lst_output = []
i = 0
while i < 2:
    if len(temp_input) > window_size:
        x_input = np.array(temp_input[1:])
        print("{} week input {}".format(i, x_input))
        x_input = x_input.reshape((1, n_features))
        yhat = model.predict(x_input)
        print("{} week output {}".format(i, yhat))
        temp_input.append(yhat[0][0])
        temp_input = temp_input[1:]
        lst_output.append(yhat[0][0])
        i = i + 1
    else:
        x_input = x_input.reshape((1, n_features))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.append(yhat[0][0])
        lst_output.append(yhat[0][0])
        i = i + 1

print(lst_output)
FP_result = pd.DataFrame(lst_output, columns=['PRED'])
# FP_result.PRED = FP_result.PRED.astype(int)
print(FP_result)


# Visualizing The Output
# future_dates = pd.date_range(start='03-10-2020', periods=4, freq='W-SUN')
# print(future_dates)
# future_dates = pd.date_range(start='3/1/2020', periods=4, freq='W-SUN')
# print(future_dates)
future_dates = pd.date_range(start=time_train['DATE'].values[-1], periods=5, freq='W-SUN')
future_dates = future_dates[1:]
future_dates = pd.DataFrame(future_dates, columns=['F_DATES'])
print(future_dates)
F_DF = pd.concat([future_dates, FP_result], axis=1)
print(F_DF)

print('Current focus:\n')
# C_df2 = pd.concat([x_train, PRED], axis=1)
# print(C_df2.tail())
print('PRED:\n', PRED)
fc = pd.concat([PRED, FP_result], axis=0).reset_index(drop=True)
print('fc\n', fc)
# #-------------------temp plot

time_d = pd.date_range(start='04/07/2019', periods=len(fc) + 2, freq='W-SUN')
time_d = pd.DataFrame(time_d, columns=['Date']).astype(str)

print('time_d\n', len(time_d)) # 52
# print(time_d)
print('x_train\n', len(x_train)) # 48
# print(x_train)
print('len_fc\n', len(fc)) # 52
# print('fc\n', fc)
print('len_series\n', len(series)) # 51

series = pd.DataFrame(series, columns=['Org_S']).astype(int)

Final_df = pd.concat([time_d, x_train, fc, series], axis=1)
Final_df.columns = ['Date', 'x_train', 'PRED_FC', 'Org_S']
# Final_df.set_index('Date', inplace=True)
print(Final_df.tail(8))

# plt.plot(Final_df['Date'], Final_df['Org_S'], label='series')
plt.plot(Final_df['Date'], Final_df['x_train'], label='x-train')
plt.plot(Final_df['Date'], Final_df['PRED_FC'], label='Forecast')
plt.title('KPL time series prediction - Actual Vs. Predicted & Forecast [ ACC:'+str(MAPE)+
          " ] Epochs: " +str(epochs)+ ' rho: ' +str(rho)+' [ LR:' + str(lr)+" ] OPT: RMSProp")
plt.xlabel("Duration")
plt.ylabel("QTY")
plt.xticks(rotation=90, size=8)
plt.subplots_adjust(bottom=0.2, left=0.06, right=0.96, top=0.9)
plt.grid(color='grey', linestyle='dotted', linewidth=0.5)
plt.grid(True, color='grey', which='major', alpha=0.5)
plt.grid(True, color='grey', which='minor', alpha=0.5)
mng = plt.get_current_fig_manager()
mng.window.state("zoomed")
plt.legend()
plt.show()

print('MAE: ', MAE)
err = 100-MAPE
print('MAPE: ', round(err), 2)
print('Accuracy: ', MAPE)

# Final_df.to_csv('F:/GangaBabu/11.01.2022/New/op.csv')

# # X = pd.DataFrame(X, index=r_df.index).astype(int)
# Final_df = pd.DataFrame(Final_df, index=time_d.index)
# print(Final_df.tail())

# #----------------------------------------------------------------
# results = results[0:]
# print(results)
# lst_output = np.array(lst_output)
# lst_output = lst_output.astype(int)
# print(F)
# print(lst_output)

# # plt.plot(pred, results)
# plt.plot(F, lst_output)
# lst_output = pd.DataFrame(lst_output)
# lst_output.to_csv('prednew.csv')
# plt.show()

# -------------------------------------------------------------------------------------------------
# # op = pd.DataFrame(results)
# # op = op.transpose()
# # a = pd.DataFrame(x_valid)
# # op.to_csv('F:/GangaBabu/15.12.2021/forecasted/op.csv')
# # a.to_csv('F:/GangaBabu/15.12.2021/forecasted/x_valid.csv')
# print(len(x_valid))
# print(len(results))

# https://towardsdatascience.com/simple-guide-to-hyperparameter-tuning-in-neural-networks-3fe03dad8594

# A_M = input('Save Param [ 1 ] \nignore [ 0 ]  \nSelect Process??\n>')
#
# if A_M == '0':
#     raise RuntimeError("Hyper parameters not saved ... '[•︵•]' ")
# elif A_M == '1':
#     print('Saving Hyper parameters ....   `\[•o•]/`')
# else:
#     pass
#
# # ##### to save the paramters and prediction resutlts in csv:
# prediction_result=pd.DataFrame(results, x_train)
# parameters_list=[window_size, batch_size, shuffle_buffer_size, lr]
# parameters_list_names=['window_size', 'batch_size', 'shuffle_buffer_size', 'learning Rate']
# parameters=pd.DataFrame(parameters_list, parameters_list_names)
# prediction_result.to_csv('prediction_result.csv')
# parameters.to_csv('paramters.csv')
# df1=pd.read_csv('paramters.csv', header=None)
# df1.columns=['parameters', 'value']
# df1.to_csv('paramaters_lists.csv', index=False)
# df = pd.read_csv('prediction_result.csv', header=None)
# df.columns = ['ACTUAL', 'PREDICTION']
# df.to_csv('pr-result.csv', index=False)
# parameters_csv=pd.read_csv('paramaters_lists.csv')
# prediction_result_csv=pd.read_csv('pr-result.csv')
# dataFrame = pd.concat([parameters_csv, prediction_result_csv], ignore_index=True)
# print(dataFrame)
# dataFrame.to_csv('prediction_values.csv')