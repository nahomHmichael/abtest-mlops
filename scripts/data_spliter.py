def data_split(df,col, cols):
  splitted_data = []
  for item in cols:
    new_data = df.loc[df[col] == item]
    splitted_data.append(new_data)

  return splitted_data