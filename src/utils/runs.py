# all area values must be converted to meters for GEE functions
# we will also report things in Ac for double-checking our math
ac_to_sqm_x = 4046.8564224
sqm_to_ac_x = 0.000247105

# Acres
treatment_size_ac = 100
treatment_size_sqm = treatment_size_ac*ac_to_sqm_x

# Each Region's Total Treatable Vegetation Acres
# from https://docs.google.com/spreadsheets/d/1Gnl6SO5kOkj4Ne1JdXzW4bp03824Zb1Cp_2HVuCY3Lk/edit#gid=0
sn_area = int(round(21979145 * ac_to_sqm_x))
sc_area = int(round(8208838 * ac_to_sqm_x))
cc_area = int(round(7368439 * ac_to_sqm_x))
nc_area = int(round(14716059 * ac_to_sqm_x))

# codified from 
# https://docs.google.com/spreadsheets/d/1Gnl6SO5kOkj4Ne1JdXzW4bp03824Zb1Cp_2HVuCY3Lk/edit#gid=0
key = {
    # Sierra Nevada
    'RunID1':{'RegionID':'SN',
              'Region Filter Name':'Sierra Nevada Region',
              'IntensityID':'500k',
              'PriorityID':'WUI',
              'acreagePerYear':210000,
              'sqmPerYear':210000*ac_to_sqm_x,
              'vecCodeLookUpList':[2,3,4],
              'vecTypeLookUpList':["year_int_6-10","year_int_11-15","year_int_1-5_16-20"]},
    
    'RunID2':{'RegionID':'SN',
              'Region Filter Name':'Sierra Nevada Region',
              'IntensityID':'500k',
              'PriorityID':'RFFC',
              'acreagePerYear':210000,
              'sqmPerYear':210000*ac_to_sqm_x,
              'vecCodeLookUpList':[1,2,3,4],
              'vecTypeLookUpList':["year_int_1-5","year_int_6-10","year_int_11-15","year_int_16-20"]},
    
    'RunID3':{'RegionID':'SN',
              'Region Filter Name':'Sierra Nevada Region',
              'IntensityID':'500k',
              'PriorityID':'Fire',
              'acreagePerYear':210000,
              'sqmPerYear':210000*ac_to_sqm_x,
              'vecCodeLookUpList':[1,2,3,4],
              'vecTypeLookUpList':["year_int_1-5","year_int_6-10","year_int_11-15","year_int_16-20"]},
    
    'RunID4':{'RegionID':'SN',
              'Region Filter Name':'Sierra Nevada Region',
              'IntensityID':'1m',
              'PriorityID':'WUI',
              'acreagePerYear':420000,
              'sqmPerYear':420000*ac_to_sqm_x,
              'vecCodeLookUpList':[2,3,4],
              'vecTypeLookUpList':["year_int_6-10","year_int_11-15","year_int_1-5_16-20"]},
    
    'RunID5':{'RegionID':'SN',
              'Region Filter Name':'Sierra Nevada Region',
              'IntensityID':'1m',
              'PriorityID':'RFFC',
              'acreagePerYear':420000,
              'sqmPerYear':420000*ac_to_sqm_x,
              'vecCodeLookUpList':[1,2,3,4],
              'vecTypeLookUpList':["year_int_1-5","year_int_6-10","year_int_11-15","year_int_16-20"]},
    
    'RunID6':{'RegionID':'SN',
              'Region Filter Name':'Sierra Nevada Region',
              'IntensityID':'1m',
              'PriorityID':'Fire',
              'acreagePerYear':420000,
              'sqmPerYear':420000*ac_to_sqm_x,
              'vecCodeLookUpList':[1,2,3,4],
              'vecTypeLookUpList':["year_int_1-5","year_int_6-10","year_int_11-15","year_int_16-20"]},
    
    'RunID7':{'RegionID':'SN',
              'Region Filter Name':'Sierra Nevada Region',
              'IntensityID':'2m',
              'PriorityID':'WUI',
              'acreagePerYear':840000,
              'sqmPerYear':840000*ac_to_sqm_x,
              'vecCodeLookUpList':[2,3,4],
              'vecTypeLookUpList':["year_int_6-10","year_int_11-15","year_int_1-5_16-20"]},
    
    'RunID8':{'RegionID':'SN',
              'Region Filter Name':'Sierra Nevada Region',
              'IntensityID':'2m',
              'PriorityID':'RFFC',
              'acreagePerYear':840000,
              'sqmPerYear':840000*ac_to_sqm_x,
              'vecCodeLookUpList':[1,2,3,4],
              'vecTypeLookUpList':["year_int_1-5","year_int_6-10","year_int_11-15","year_int_16-20"]},
    
    'RunID9':{'RegionID':'SN',
              'Region Filter Name':'Sierra Nevada Region',
              'IntensityID':'2m',
              'PriorityID':'Fire',
              'acreagePerYear':840000,
              'sqmPerYear':840000*ac_to_sqm_x,
              'vecCodeLookUpList':[1,2,3,4],
              'vecTypeLookUpList':["year_int_1-5","year_int_6-10","year_int_11-15","year_int_16-20"]},
    
    # North Coast
    'RunID10':{'RegionID':'NC',
               'Region Filter Name':'North Coast Region',
               'IntensityID':'500k',
               'PriorityID':'WUI',
               'acreagePerYear':140000,
               'sqmPerYear':140000*ac_to_sqm_x,
               'vecCodeLookUpList':[2,3,4],
              'vecTypeLookUpList':["year_int_6-10","year_int_11-15","year_int_1-5_16-20"]},
    
    'RunID11':{'RegionID':'NC',
               'Region Filter Name':'North Coast Region',
               'IntensityID':'500k',
               'PriorityID':'RFFC',
               'acreagePerYear':140000,
               'sqmPerYear':140000*ac_to_sqm_x,
               'vecCodeLookUpList':[1,2,3,4],
              'vecTypeLookUpList':["year_int_1-5","year_int_6-10","year_int_11-15","year_int_16-20"]},
    
    'RunID12':{'RegionID':'NC',
               'Region Filter Name':'North Coast Region',
               'IntensityID':'500k',
               'PriorityID':'Fire',
               'acreagePerYear':140000,
               'sqmPerYear':140000*ac_to_sqm_x,
               'vecCodeLookUpList':[1,2,3,4],
              'vecTypeLookUpList':["year_int_1-5","year_int_6-10","year_int_11-15","year_int_16-20"]},
    
    'RunID13':{'RegionID':'NC',
               'Region Filter Name':'North Coast Region',
               'IntensityID':'1m',
               'PriorityID':'WUI',
               'acreagePerYear':280000,
               'sqmPerYear':280000*ac_to_sqm_x,
               'vecCodeLookUpList':[2,3,4],
              'vecTypeLookUpList':["year_int_6-10","year_int_11-15","year_int_1-5_16-20"]},
    
    'RunID14':{'RegionID':'NC',
               'Region Filter Name':'North Coast Region',
               'IntensityID':'1m',
               'PriorityID':'RFFC',
               'acreagePerYear':280000,
               'sqmPerYear':280000*ac_to_sqm_x,
               'vecCodeLookUpList':[1,2,3,4],
              'vecTypeLookUpList':["year_int_1-5","year_int_6-10","year_int_11-15","year_int_16-20"]},
    
    'RunID15':{'RegionID':'NC',
               'Region Filter Name':'North Coast Region',
               'IntensityID':'1m',
               'PriorityID':'Fire',
               'acreagePerYear':280000,
               'sqmPerYear':280000*ac_to_sqm_x,
               'vecCodeLookUpList':[1,2,3,4],
              'vecTypeLookUpList':["year_int_1-5","year_int_6-10","year_int_11-15","year_int_16-20"]},
    
    'RunID16':{'RegionID':'NC',
               'Region Filter Name':'North Coast Region',
               'IntensityID':'2m',
               'PriorityID':'WUI',
               'acreagePerYear':560000,
               'sqmPerYear':560000*ac_to_sqm_x,
               'vecCodeLookUpList':[2,3,4],
              'vecTypeLookUpList':["year_int_6-10","year_int_11-15","year_int_1-5_16-20"]},
    
    'RunID17':{'RegionID':'NC',
               'Region Filter Name':'North Coast Region',
               'IntensityID':'2m',
               'PriorityID':'RFFC',
               'acreagePerYear':560000,
               'sqmPerYear':560000*ac_to_sqm_x,
               'vecCodeLookUpList':[1,2,3,4],
              'vecTypeLookUpList':["year_int_1-5","year_int_6-10","year_int_11-15","year_int_16-20"]},
    
    'RunID18':{'RegionID':'NC',
               'Region Filter Name':'North Coast Region',
               'IntensityID':'2m',
               'PriorityID':'Fire',
               'acreagePerYear':560000,
               'sqmPerYear':560000*ac_to_sqm_x,
               'vecCodeLookUpList':[1,2,3,4],
              'vecTypeLookUpList':["year_int_1-5","year_int_6-10","year_int_11-15","year_int_16-20"]},
    
    # South Coast
    'RunID19':{'RegionID':'SC',
              'Region Filter Name':'South Coast Region',
              'IntensityID':'500k',
              'PriorityID':'WUI',
              'RoadGrids': 'projects/mas-gee/assets/scTxGridRoad',
              'acreagePerYear':80000,
              'sqmPerYear':80000*ac_to_sqm_x,
              'vecCodeLookUpList':[2,3,4],
              'vecTypeLookUpList':["year_int_6-10","year_int_11-15","year_int_1-5_16-20"]},
    
    'RunID20':{'RegionID':'SC',
              'Region Filter Name':'South Coast Region',
              'IntensityID':'500k',
              'PriorityID':'RFFC',
              'RoadGrids': 'projects/mas-gee/assets/scTxGridRoad',
              'acreagePerYear':80000,
              'sqmPerYear':80000*ac_to_sqm_x,
              'vecCodeLookUpList':[1,2,3,4],
              'vecTypeLookUpList':["year_int_1-5","year_int_6-10","year_int_11-15","year_int_16-20"]},
    
    'RunID21':{'RegionID':'SC',
              'Region Filter Name':'South Coast Region',
              'IntensityID':'500k',
              'PriorityID':'Fire',
              'RoadGrids': 'projects/mas-gee/assets/scTxGridRoad',
              'acreagePerYear':80000,
              'sqmPerYear':80000*ac_to_sqm_x,
              'vecCodeLookUpList':[1,2,3,4],
              'vecTypeLookUpList':["year_int_1-5","year_int_6-10","year_int_11-15","year_int_16-20"]},
    
    'RunID22':{'RegionID':'SC',
              'Region Filter Name':'South Coast Region',
              'IntensityID':'1m',
              'PriorityID':'WUI',
              'RoadGrids': 'projects/mas-gee/assets/scTxGridRoad',
              'acreagePerYear':160000,
              'sqmPerYear':160000*ac_to_sqm_x,
              'vecCodeLookUpList':[2,3,4],
              'vecTypeLookUpList':["year_int_6-10","year_int_11-15","year_int_1-5_16-20"]},
    
    'RunID23':{'RegionID':'SC',
              'Region Filter Name':'South Coast Region',
              'IntensityID':'1m',
              'PriorityID':'RFFC',
              'RoadGrids': 'projects/mas-gee/assets/scTxGridRoad',
              'acreagePerYear':160000,
              'sqmPerYear':160000*ac_to_sqm_x,
              'vecCodeLookUpList':[1,2,3,4],
              'vecTypeLookUpList':["year_int_1-5","year_int_6-10","year_int_11-15","year_int_16-20"]},
    
    'RunID24':{'RegionID':'SC',
              'Region Filter Name':'South Coast Region',
              'IntensityID':'1m',
              'PriorityID':'Fire',
              'RoadGrids': 'projects/mas-gee/assets/scTxGridRoad',
              'acreagePerYear':160000,
              'sqmPerYear':160000*ac_to_sqm_x,
              'vecCodeLookUpList':[1,2,3,4],
              'vecTypeLookUpList':["year_int_1-5","year_int_6-10","year_int_11-15","year_int_16-20"]},
    
    'RunID25':{'RegionID':'SC',
              'Region Filter Name':'South Coast Region',
              'IntensityID':'2m',
              'PriorityID':'WUI',
              'RoadGrids': 'projects/mas-gee/assets/scTxGridRoad',
              'acreagePerYear':368000,
              'sqmPerYear':368000*ac_to_sqm_x,
              'vecCodeLookUpList':[2,3,4],
              'vecTypeLookUpList':["year_int_6-10","year_int_11-15","year_int_1-5_16-20"]},
    
    'RunID26':{'RegionID':'SC',
              'Region Filter Name':'South Coast Region',
              'IntensityID':'2m',
              'PriorityID':'RFFC',
              'RoadGrids': 'projects/mas-gee/assets/scTxGridRoad',
              'acreagePerYear':368000,
              'sqmPerYear':368000*ac_to_sqm_x,
              'vecCodeLookUpList':[1,2,3,4],
              'vecTypeLookUpList':["year_int_1-5","year_int_6-10","year_int_11-15","year_int_16-20"]},
    
    'RunID27':{'RegionID':'SC',
              'Region Filter Name':'South Coast Region',
              'IntensityID':'2m',
              'PriorityID':'Fire',
              'RoadGrids': 'projects/mas-gee/assets/scTxGridRoad',
              'acreagePerYear':368000,
              'sqmPerYear':368000*ac_to_sqm_x,
              'vecCodeLookUpList':[1,2,3,4],
              'vecTypeLookUpList':["year_int_1-5","year_int_6-10","year_int_11-15","year_int_16-20"]},
    
    # Central Coast
    'RunID28':{'RegionID':'CC',
              'Region Filter Name':'Central Coast Region',
              'IntensityID':'500k',
              'PriorityID':'WUI',
              'RoadGrids': 'projects/mas-gee/assets/ccTxGridRoad',
              'acreagePerYear':70000,
              'sqmPerYear':70000*ac_to_sqm_x,
              'vecCodeLookUpList':[2,3,4],
              'vecTypeLookUpList':["year_int_6-10","year_int_11-15","year_int_1-5_16-20"]},
    
    'RunID29':{'RegionID':'CC',
              'Region Filter Name':'Central Coast Region',
              'IntensityID':'500k',
              'PriorityID':'RFFC',
              'RoadGrids': 'projects/mas-gee/assets/ccTxGridRoad',
              'acreagePerYear':70000,
              'sqmPerYear':70000*ac_to_sqm_x,
              'vecCodeLookUpList':[1,2,3,4],
              'vecTypeLookUpList':["year_int_1-5","year_int_6-10","year_int_11-15","year_int_16-20"]},
    
    'RunID30':{'RegionID':'CC',
              'Region Filter Name':'Central Coast Region',
              'IntensityID':'500k',
              'PriorityID':'Fire',
              'RoadGrids': 'projects/mas-gee/assets/ccTxGridRoad',
              'acreagePerYear':70000,
              'sqmPerYear':70000*ac_to_sqm_x,
              'vecCodeLookUpList':[1,2,3,4],
              'vecTypeLookUpList':["year_int_1-5","year_int_6-10","year_int_11-15","year_int_16-20"]},
    
    'RunID31':{'RegionID':'CC',
              'Region Filter Name':'Central Coast Region',
              'IntensityID':'1m',
              'PriorityID':'WUI',
              'RoadGrids': 'projects/mas-gee/assets/ccTxGridRoad',
              'acreagePerYear':140000,
              'sqmPerYear':140000*ac_to_sqm_x,
              'vecCodeLookUpList':[2,3,4],
              'vecTypeLookUpList':["year_int_6-10","year_int_11-15","year_int_1-5_16-20"]},
    
    'RunID32':{'RegionID':'CC',
              'Region Filter Name':'Central Coast Region',
              'IntensityID':'1m',
              'PriorityID':'RFFC',
              'RoadGrids': 'projects/mas-gee/assets/ccTxGridRoad',
              'acreagePerYear':140000,
              'sqmPerYear':140000*ac_to_sqm_x,
              'vecCodeLookUpList':[1,2,3,4],
              'vecTypeLookUpList':["year_int_1-5","year_int_6-10","year_int_11-15","year_int_16-20"]},
    
    'RunID33':{'RegionID':'CC',
              'Region Filter Name':'Central Coast Region',
              'IntensityID':'1m',
              'PriorityID':'Fire',
              'RoadGrids': 'projects/mas-gee/assets/ccTxGridRoad',
              'acreagePerYear':140000,
              'sqmPerYear':140000*ac_to_sqm_x,
              'vecCodeLookUpList':[1,2,3,4],
              'vecTypeLookUpList':["year_int_1-5","year_int_6-10","year_int_11-15","year_int_16-20"]},
    
    'RunID34':{'RegionID':'CC',
              'Region Filter Name':'Central Coast Region',
              'IntensityID':'2m',
              'PriorityID':'WUI',
              'RoadGrids': 'projects/mas-gee/assets/ccTxGridRoad',
              'acreagePerYear':322000,
              'sqmPerYear':322000*ac_to_sqm_x,
              'vecCodeLookUpList':[2,3,4],
              'vecTypeLookUpList':["year_int_6-10","year_int_11-15","year_int_1-5_16-20"]},
    
    'RunID35':{'RegionID':'CC',
              'Region Filter Name':'Central Coast Region',
              'IntensityID':'2m',
              'PriorityID':'RFFC',
              'RoadGrids': 'projects/mas-gee/assets/ccTxGridRoad',
              'acreagePerYear':322000,
              'sqmPerYear':322000*ac_to_sqm_x,
              'vecCodeLookUpList':[1,2,3,4],
              'vecTypeLookUpList':["year_int_1-5","year_int_6-10","year_int_11-15","year_int_16-20"]},
    
    'RunID36':{'RegionID':'CC',
              'Region Filter Name':'Central Coast Region',
              'IntensityID':'2m',
              'PriorityID':'Fire',
              'RoadGrids': 'projects/mas-gee/assets/ccTxGridRoad',
              'acreagePerYear':322000,
              'sqmPerYear':322000*ac_to_sqm_x,
              'vecCodeLookUpList':[1,2,3,4],
              'vecTypeLookUpList':["year_int_1-5","year_int_6-10","year_int_11-15","year_int_16-20"]},
    
    }