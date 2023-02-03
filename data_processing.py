import random
import re
import pandas as pd

# initial data import

data = pd.read_csv('datasets/flats_dataset.csv')
data.drop('storey_range', axis='columns', inplace=True)
data.drop('street_name', axis='columns', inplace=True)
data.drop('lease_commence_date', axis='columns', inplace=True)


# processing number of rooms and types of rooms columns, as well as block (sometimes it has a single string sign at
# the end)

def block_process(number):
    result = [x for x in re.sub(r'[a-zA-z]', '', number)]
    return int(''.join(result))


data = data.rename(columns={'flat_type': 'no_of_rooms'})
data['no_of_rooms'] = [x[0] for x in data['no_of_rooms']]
data['no_of_rooms'] = data['no_of_rooms'].apply(lambda x: int(x) if x.isnumeric() else 0)
data['block'] = data['block'].apply(lambda x: block_process(x) if isinstance(x, str) else x)
data['is_ex'] = [1 if x == 0 else 0 for x in data['no_of_rooms']]


# processing remaining lease - it's given in years, so I will leave it that way and take full years into account only,
# so any additional months will be rounded to full years (if it's above 6 months, it rounds up)

def lease_process(lease):
    lease_reg = re.compile(r'\d{2}')
    result = [int(x) for x in lease_reg.findall(lease)]
    if len(result) > 1:
        if result[1] > 6:
            return result[0] + 1
    return result[0]


data['remaining_lease'] = data['remaining_lease'].apply(lambda x: random.randint(60, 80) if pd.isna(x) else x)
data['remaining_lease'] = data['remaining_lease'].apply(lambda x: lease_process(x) if isinstance(x, str) else x)


# processing date - the year and month should be seperated

def date_process(date, desired_part):
    date_reg = re.compile(r'\d+')
    result = [int(x) for x in date_reg.findall(date)]
    if desired_part == 'year':
        return result[0]
    else:
        return result[1]


data['year'] = data['month'].apply(lambda x: date_process(x, 'year'))
data['month'] = data['month'].apply(lambda x: date_process(x, 'month'))

data_order = ['resale_price', 'year', 'month', 'town', 'no_of_rooms', 'flat_model', 'is_ex', 'block', 'floor_area_sqm',
              'remaining_lease']
data = data.reindex(data_order, axis=1)

# getting dummies for different towns and flat models
data = pd.get_dummies(data, columns=['town', 'flat_model'])

# saving final dataset with processed flats info to .csv file
data.to_csv('datasets\\flats_processed.csv', index=False)
print(data)
