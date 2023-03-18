import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
import warnings
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'
import tensorflow as tf
from tensorflow import keras
warnings.simplefilter(action='ignore', category=FutureWarning)
from datetime import datetime
#import mpld3
import datetime


koneksi_mongodb = "mongodb+srv://midya123:1234567890@cluster0.wsuei7i.mongodb.net/test"
cluster = MongoClient(koneksi_mongodb)
db = cluster['iklim'] 
collection = db['dataiklim2'] 

data = collection.find()
df = pd.json_normalize(data)
dfIklim = pd.DataFrame(df)
# menghapus data id
df = dfIklim.drop(columns = ['_id'])
#mengubah tipe data tanggal
df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d-%m-%Y') 

df_geo = pd.DataFrame({'lat': [-8.75000], 'lon': [115.17000]})

option = st.sidebar.selectbox(
    'Silakan pilih:',
    ('Home','Prediksi Matahari','Prediksi Angin','Estimasi')
)

if option == 'Home' or option == '':
    st.title(""":blue[Prediksi dan Estimasi ]""")
    st.write("""### Penyinaran Matahari:mostly_sunny: & Kecepatan Angin:cyclone:""") #menampilkan halaman utama
    st.text("Lokasi Pengamatan")
    st.map(df_geo)
    st.text("Stasiun Metereologi Kelas I Ngurah Rai")
elif option == 'Prediksi Matahari':
    st.title("Prediksi Penyinaran Matahari") #menampilkan judul halaman 

    #download template
    st.write("- prediksi yang dilakukan membutuhkan data sebanyak jumlah prediksi yang diinginkan ditambah 1 bulan")
    st.write("- untuk memulai proses prediksi, silahkan download template terlebih dahulu")
    st.write("")
    st.text("") 
    with open("assets/template_matahari.csv") as file:
        btn = st.download_button(
            label="Download template",
            data=file,
            file_name="template_matahari.csv",
            mime="text/csv"
        )

    #upload file
    st.text("")
    st.text("")

    #input year 
    year = st.number_input('Masukkan Tahun Prediksi', key = "a", min_value =2000, max_value=None)

    uploaded_file = st.file_uploader("silahkan upload template yang telah memuat data yang diperlukan")
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        st.write("Data telah berhasil di load, berikut adalah data yang digunakan untuk proses prediksi") 
        st.write(dataframe) 

        #dataframe['Tanggal'] = pd.to_datetime(dataframe['Tanggal'], format='%d-%m-%Y') 
        dataframe.columns = ['Tanggal','suhu_max' ,'suhu_avg','hujan','matahari']
        dataframe = dataframe.set_index('Tanggal')
        df2 = dataframe.dropna()
        #df_angin = df['ff_avg']

        #plot df asli
        st.write("Visualisasi Data")
        st.line_chart(df2)

        #creating lag 1 and rolling window feature
        df2['lag_1'] = df2['matahari'].shift(1)
        rolling = df2['matahari'].rolling(window=30)
        rolling_sum = rolling.sum()
        df2 = pd.merge(df2,rolling_sum,on='Tanggal')
        df2 = df2.dropna()
        
        #membagi x dan y
        #from sklearn.model_selection import train_test_split
        X= df2.drop('matahari_x', axis=1)
        y= df2['matahari_x']
        
        #minmaxscaler
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        #scaled =pd.DataFrame(scaled)

        scaler2 = MinMaxScaler()
        y = y.to_numpy()
        test1 = y.reshape(-1,1)
        y = scaler2.fit_transform(test1)

        #X= X.to_numpy()
        n_features = 5
        n_seq = 1
        n_steps = 1
        X = X.reshape((X.shape[0], n_seq, n_steps, n_features))

        #Loading Model
        model = keras.models.load_model('assets/new_5feature_cnngru.h5')


        #Testing Model
        predictions = model.predict(X).flatten()


        #save prediction result
        pred = predictions.reshape(-1,1)
        pred = scaler2.inverse_transform(pred)
        
        #menyimpan data prediksi matahari
        pred = pd.DataFrame(pred, columns =['prediksi (jam)'])

        year = year
        # loop through all days of the year
        dates = pd.date_range(start=datetime.date(year, 1, 1), end=datetime.date(year, 12, 31))
        dates = pd.DataFrame({'Tanggal': dates})

        df_pred =pred.join(dates)
        df_pred = df_pred.set_index('Tanggal')

        st.write('\n')
        st.write('\n')
        st.write('\n')
        #menampilkan hasil prediksi dalam bentuk tabel
        st.header('Prediksi rata-rata harian dengan CNN-GRU')


        #menampilkan hasil prediksi
        st.write(df_pred)
        st.write('\n')
        st.subheader('Visualisasi Prediksi Intensitas Penyinaran Matahari')

        st.line_chart(df_pred)

        #@st.cache_data
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_pred)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='result_matahari.csv',
            mime='text/csv',
        ) 

    else:
        st.text("Silahkan Upload file")    

elif option == 'Prediksi Angin':
    st.title("Prediksi Kecepatan Angin") #menampilkan judul halaman 

    #download template
    st.write("- prediksi yang dilakukan membutuhkan data sebanyak jumlah prediksi yang diinginkan ditambah 1 bulan")
    st.write("- untuk memulai proses prediksi, silahkan download template terlebih dahulu")
    st.write("")
    st.text("") 
    with open("assets/template_angin.csv") as file:
        btn = st.download_button(
            label="Download template",
            data=file,
            file_name="template_angin.csv",
            mime="text/csv"
        )

    #upload file
    st.text("")
    st.text("")

    #input year 
    year = st.number_input('Masukkan Tahun Prediksi', key = "a", min_value =2000, max_value=None)


    uploaded_file = st.file_uploader("silahkan upload template yang telah memuat data yang diperlukan")
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        st.write("Data telah berhasil di load, berikut adalah data yang digunakan untuk proses prediksi") 
        st.write(dataframe) 

        #dataframe['Tanggal'] = pd.to_datetime(dataframe['Tanggal'], format='%d-%m-%Y') 
        dataframe.columns = ['Tanggal','suhu_max' ,'matahari','angin_max','angin']
        dataframe = dataframe.set_index('Tanggal')
        df2 = dataframe.dropna()
        #df_angin = df['ff_avg']

        #plot df asli
        st.write("Visualisasi Data")
        st.line_chart(df2)

        #creating lag 1 and rolling window feature
        df2['lag_1'] = df2['angin'].shift(1)
        rolling = df2['angin'].rolling(window=30)
        rolling_sum = rolling.sum()
        df2 = pd.merge(df2,rolling_sum,on='Tanggal')
        df2 = df2.dropna()
        
        #membagi x dan y
        #from sklearn.model_selection import train_test_split
        X= df2.drop('angin_x', axis=1)
        y= df2['angin_x']
        
        #minmaxscaler
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        #scaled =pd.DataFrame(scaled)

        scaler2 = MinMaxScaler()
        y = y.to_numpy()
        test1 = y.reshape(-1,1)
        y = scaler2.fit_transform(test1)

        #X= X.to_numpy()
        n_features = 5
        n_seq = 1
        n_steps = 1
        X = X.reshape((X.shape[0], n_seq, n_steps, n_features))

        #Loading Model
        model = keras.models.load_model('assets/fulldata_cnn_bilstm.h5')
        #Testing Model
        predictions = model.predict(X).flatten()


        #save prediction result
        pred = predictions.reshape(-1,1)
        pred = scaler2.inverse_transform(pred)
        
        st.write("\n")
        st.write("\n")
        st.write("\n")
        #menampilkan hasil prediksi dalam bentuk tabel
        st.header('Prediksi rata-rata harian CNN-BiLSTM')

        #menyimpan data prediksi
        pred = pd.DataFrame(pred, columns =['prediksi (m/s)'])

        year = year
        # loop through all days of the year
        dates = pd.date_range(start=datetime.date(year, 1, 1), end=datetime.date(year, 12, 31))
        dates = pd.DataFrame({'Tanggal': dates})

        df_pred =pred.join(dates)
        df_pred = df_pred.set_index('Tanggal')
        # df_pred = df_pred.iloc[:,[1,0]]
        # df_pred = df_pred.dropna()
        # df_pred.columns = ['tanggal','prediksi']
        st.write(df_pred)

        # #menampilkan hasil prediksi
        st.write('### Visualisasi Prediksi Kecepatan Angin Rata-Rata(m/s)')

        # chart_width = st.expander(label="chart width").slider("", 10, 32, 14)

        # fig2 = plt.figure(figsize = (chart_width,10))
        # plt.plot(df_pred);

        # st.write(fig2)
        # plt.xlabel(df_pred['prediksi'])
        # plt.ylabel(df_pred['Tanggal'])
        # #plt.legend()
        # #st.pyplot(fig2)

        st.line_chart(df_pred)

        #@st.cache_data
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_pred)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='result_angin.csv',
            mime='text/csv',
        ) 

    else:
        st.text("") 

# elif option == 'Exploratory Data Analysis':
#     st.write("""## Visualisasi Data yang Digunakan Pada Proses Training Model""") #menampilkan judul halaman 

#     #membuat variabel chart data yang berisi data dari dataframe
#     #data berupa angka acak yang di-generate menggunakan numpy
#     #data terdiri dari 2 kolom dan 20 baris
#     chart_data1 = pd.DataFrame(df, 
#         columns=['ss'])
#     chart_data2 = pd.DataFrame(df, 
#         columns=['ff_avg']
#     )

#     st.write("### Keterangan")
#     st.text("ss = Penyinaran Matahari")
#     st.text("ff_avg = Rata-Rata Kecepatan Angin")
#     #menampilkan data dalam bentuk chart
#     st.line_chart(chart_data1)
#     st.line_chart(chart_data2)
    
    

elif option == 'Estimasi':
    st.write("""## Estimasi Kecepatan Angin""") #menampilkan judul halaman 
    a = st.number_input('Masukkan Suhu Maksimum', key = "a", min_value =26, max_value=40)
    b = st.number_input('Masukkan Intensitas Penyinaran Matahari', key = "b", max_value=12)
    c = st.number_input('Masukkan Maksimum Kecepatan Angin', key = "c", max_value=50)
    
    import pickle
    regression = pickle.load(open("assets/estimasi_angin.pickle", "rb"))

    estimation = regression.predict([[a, b, c]])
    estimation = estimation[0]
    result = round(estimation, 2)
    st.text("Hasil Estimasi Kecepatan Angin hari berikutnya (m/s) :")
    if a == 0 or b == 0 or c==0:
        st.text("0")
    else:
        st.text(result)


    st.write(""" """) 
    #estimasi matahari dengan suhu maksimum, suhu rata-rata, curah hujan
    st.write("""## Estimasi Intensitas Penyinaran Matahari""") 
    d = st.number_input('Masukkan Suhu Maksimum', key = "d", min_value =26, max_value=40)
    e = st.number_input('Masukkan Suhu Rata-Rata', key = "e", min_value =26, max_value=38)
    f = st.number_input('Masukkan Curah Hujan', key = "f")
    
    import pickle
    regression_sun = pickle.load(open("assets/estimasi_matahari.pickle", "rb"))

    estimation_sun = regression_sun.predict([[d, e, f]])
    estimation_sun = estimation_sun[0]
    result_sun = round(estimation_sun, 2)
    st.text("Hasil Estimasi Penyinaran Matahari hari berikutnya (jam) :")
    if d == 0 or e == 0:
        st.text("0") 
    else:
        st.text(result_sun)
