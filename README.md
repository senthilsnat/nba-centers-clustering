# nba-centers-clustering

Files relevant to clustering of starting centers and 2D visualization of the clustering as described in http://fansided.com/2016/10/25/nylon-calculus-referendum-center-position/

Please credit Senthil S. Natarajan if using or modifying the work contained herein, or contact via Twitter @SENTH1S

Files Overview:
---
1. **"centers_data.csv":** CSV file containing original data on 30 starting centers pulled from BBALL-Reference

2. **"centers_data_wvorp.csv":** CSV file containing original data on 30 starting centers pulled from BBALL-Reference including VORP for each center (necessary for feature selection experiment)

3. **"kmeans.py":** main python script for K-Means clustering of 30 starting centers, including dimensionality reduction and 2D visualization of clustering

4. **"fsel.py":** python script for feature selection experiment to identify which statistical categories best correlate to VORP from data on 30 starting centers
