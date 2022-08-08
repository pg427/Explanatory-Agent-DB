"""
WHAT: Program that loads the local similarity information from bmt tree
      Restructured Prateek's Jupyter notebook into python script
      Populates xARA_LocalSimNodes and xARA_LocalSimPredicates
WHY: To calculate and load local similarity between nodes and predicates
ASSUMES:
FUTURE IMPROVEMENTS:
WHO: Manil 09-21-2021
"""
from bmt import Toolkit
import numpy as np
import pandas as pd
import datetime
import logging
from lib.clsSQLiteDatabase import clsSQLiteDatabase
current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# Log information
log_file = '../log/load_local_similarity_nodes_predicates_' + current_time + '.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

# Initialize database object
db_path = "../data/xARA_DB.db"
db = clsSQLiteDatabase(db_path)

# KP Inventory excel file
inventory_file = '../data/kp_inventory/KP_Inventory_file.xlsx'
#inventory_file = '../data/kp_inventory/KP_Inventory_Reduced.xlsx'
# kp_inventory_file = '../data/kp_inventory/KP_Inventory_All.xlsx'

# Initialize toolkit object
t = Toolkit()


def get_ancestors(arr):
    '''
    Recursively gets the parents of the given entity in the bmt tree
    :param arr:
    :return:
    '''
    for x in arr[0]:
        if t.get_parent(x):
            arr.insert(0, [t.get_parent(x)])
            get_ancestors(arr)
        else:
            return arr


def get_descendents(arr):
    '''
    Gets the descendents of the given entity from bmt tree
    :param arr:
    :return:
    '''
    arr.append([])
    i = 0
    for x in arr[-2]:
        if t.get_children(x):
            arr[-1].extend(t.get_children(x))
            i = 1
    if i == 0:
        return arr
    else:
        get_descendents(arr)


def get_tree(Entity):
    '''
    Constructs the tree structure for the given entity using get ancestor and get descendents
    :param Entity:
    :return:
    '''
    Entity_arr = []
    Entity_arr_nod = []
    Entity_arr_chk = []
    Entity_arr.append([])
    Entity_arr[0].append(Entity)
    get_ancestors(Entity_arr)
    get_descendents(Entity_arr)
    Entity_arr = Entity_arr[:-1]
    for x in Entity_arr:
        for y in x.copy():
            if Entity_arr_chk == []:
                Entity_arr_chk.append(y)
            else:
                if y in Entity_arr_chk:
                    x.remove(y)
                else:
                    Entity_arr_chk.append(y)
        if x != []:
            Entity_arr_nod.append(x)

    return Entity_arr_nod


def index_2d(myList, v):
    '''
    Returns the index of the given value v in the 2d list myList
    :param myList:
    :param v:
    :return:
    '''
    for i, x in enumerate(myList):
        if v in x:
            return (i, x.index(v))

# Load the kp inventory information into pandas dataframe
df = pd.read_excel(inventory_file)

subject_categories = list(df['subject'])
object_categories = list(df['object'])
all_categories = []
all_categories.extend(subject_categories)
all_categories.extend(object_categories)
all_categories = list(set(all_categories))
all_categories_new = []

# Changing the name of categories so it is same has standard prefix and style "biolink:"
for g in all_categories:
    g_temp = g.replace('biolink:', '')
    g_temp2 = g_temp[0].lower()
    for i in range(1, len(g_temp)-1):
        if g_temp[i].isupper():
            if g_temp[i-1].islower() or g_temp[i+1].islower():
                g_temp2 = g_temp2 + " "
            g_temp2 = g_temp2 + g_temp[i].lower()
        else:
            g_temp2 = g_temp2 + g_temp[i]
    g_temp2 = g_temp2 + g_temp[len(g_temp)-1].lower()
    all_categories_new.append(g_temp2)

logging.info('Node categories similarity calculated')
print('Node categories similarity calculated')


# Changing the name of predicate so it is same has standard prefix and style "biolink:"
predicates = list(df['predicate'])
predicates_new = []
for g in predicates:
    g_temp = g.replace('biolink:', '')
    predicates_new.append(g_temp.replace('_', ' '))
predicates_distinct = list(set(predicates_new))

# Calculate the tree structure for the predicates including the similarity value
localsim_predicates_map = []
for idx, x in enumerate(predicates_distinct):
    nc_tree = get_tree(x)
    predicate_position = index_2d(nc_tree, x)[0]
    nc_tree_levels = max(predicate_position-0, len(nc_tree)-predicate_position)
    localsim_predicates_map.append([])
    for idy, y in enumerate(predicates_distinct):
        if x==y:
            localsim_predicates_map[idx].append(1)
        else:
            temp_pos = index_2d(nc_tree, y)
            if temp_pos==None:
                localsim_predicates_map[idx].append(0)
            else:
                if temp_pos[0] - predicate_position > 0:
                    tmp = 1 - ((np.abs(temp_pos[0] - predicate_position)-0.1)* (1/(nc_tree_levels+1)))
                else:
                    tmp = 1 - (np.abs(temp_pos[0] - predicate_position)* (1/(nc_tree_levels+1)))
                localsim_predicates_map[idx].append(tmp)

logging.info('Predicate similarity calculated')
print('Predicates similarity calculated')

# Calculate the tree structure for the node categories including the similarity value
localsim_categories_map = []
for idx, x in enumerate(all_categories_new):
    nc_tree = get_tree(x)
    category_position = index_2d(nc_tree, x)[0]
    nc_tree_levels = max(category_position-0, len(nc_tree)-category_position)
    localsim_categories_map.append([])
    for idy, y in enumerate(all_categories_new):
        if x==y:
            localsim_categories_map[idx].append(1)
        else:
            temp_pos = index_2d(nc_tree, y)
            if temp_pos==None:
                localsim_categories_map[idx].append(0)
            else:
                tmp = 1 - (np.abs(temp_pos[0] - category_position) * (1/(nc_tree_levels+1)))
                localsim_categories_map[idx].append(tmp)

# These conflation cases defy the local similarity score calculated as they are special cases of high similarity
conflation_cases = [["gene", "protein", 0.98],
                    ["protein", "gene", 0.98]]

for idx, x in enumerate(all_categories_new):
    for idy, y in enumerate(all_categories_new):
        for idl, l in enumerate(conflation_cases):
            if x==l[0] and y==l[1]:
                localsim_categories_map[idx][idy] = l[2]

logging.info('Conflation rules applied')
print('Conflation rules applied')

try:
    db.start_connection()
    #print(db.get_data("select count(*) from xara_caseproblems;"))
    # Create table for local similarity of predicates if they do not exist
    ct_query = """
                CREATE TABLE IF NOT EXISTS xARA_LocalSimPredicates 
                (
                NEW_CASE_PREDICTATE TEXT
                , CANDIDATE_CASE_PREDICATE TEXT
                , SIMILARITY_SCORE integer
                )
                """
    db.run_query(ct_query)

    # Deleting from the table
    db.run_query("DELETE FROM xARA_LocalSimPredicates")

    # Insert into the database table
    for idx, x in enumerate(localsim_predicates_map):
        for idy, y in enumerate(x):
            pdtx = 'biolink:' + predicates_distinct[idx].replace(" ", "_").lower()
            pdty = 'biolink:' +predicates_distinct[idy].replace(" ", "_").lower()
            varlist = [pdtx, pdty, y]

            var_string = ', '.join('?' * len(varlist))
            query_string = 'INSERT INTO xARA_LocalSimPredicates VALUES (%s);' % var_string

            db.run_query_with_params(query_string, varlist)

    logging.info('Data for table: xARA_LocalSimPredicates loaded')
    print('Data for table: xARA_LocalSimPredicates loaded')

    # Create table for local similarity of nodes if they do not exist
    ct_query = """
                CREATE TABLE IF NOT EXISTS xARA_LocalSimNodes 
                (   NEW_CASE_NODE TEXT
                    , CANDIDATE_CASE_NODE TEXT
                    , SIMILARITY_SCORE integer
                )
                """
    db.run_query(ct_query)

    # Deleting from the table
    db.run_query("DELETE FROM xARA_LocalSimNodes")

    # Insert into the database table
    for idx, x in enumerate(localsim_categories_map):
        for idy, y in enumerate(x):
            acatnx = 'biolink:' + all_categories_new[idx].title().replace(" ", "")
            acatny = 'biolink:' + all_categories_new[idy].title().replace(" ", "")
            varlist = [acatnx, acatny, y]
            var_string = ', '.join('?' * len(varlist))
            query_string = 'INSERT INTO xARA_LocalSimNodes VALUES (%s);' % var_string
            db.run_query_with_params(query_string, varlist)

    db.run_query("commit")

    logging.info('Data for table: xARA_LocalSimNodes loaded')
    print('Data for table: xARA_LocalSimNodes loaded')

    logging.info('Program completed successfully')
    print('Program completed successfully')

except Exception as e:
    print("Error! : ", str(e))
    raise e

finally:
    db.end_connection()
