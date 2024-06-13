# import pandas as pd
# import plotly.express as px

# # Sample DataFrame with latitude, longitude, and time in seconds
# data = {
#     'latitude': [40.7128, 34.0522, 37.7749],  # Example latitudes
#     'longitude': [-74.0060, -118.2437, -122.4194],  # Example longitudes
#     'time_seconds': [0, 3600, 7200]  # Example times in seconds
# }

# df = pd.DataFrame(data)

# # Plotting using Plotly Express with a time slider
# fig = px.scatter_geo(df, lat='latitude', lon='longitude', animation_frame='time_seconds',
#                      title='Location Over Time', scope='world')
# fig.update_geos(projection_type="natural earth")
# fig.show()
print(-200%360)