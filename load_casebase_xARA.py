"""
WHAT: Program that loads CaseBase database xARA from KP Inventory excel file.
WHY: To load xARA_DB with case problems and case solutions
ASSUMES:
FUTURE IMPROVEMENTS:
WHO: Manil 2021-05-26
     Manil 2021-09-15 Performance enhancements with the use of dataframes, multithreading and use of local sim node and
                      pred tables instead of BMT tree traversal
"""
import datetime
import pickle
import os
current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
import logging

# Log information
log_file = '../log/load_casebase_xARA_' + current_time + '.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

import numpy as np
import pandas as pd
from lib.clsSQLiteDatabase import clsSQLiteDatabase
from sqlalchemy import create_engine
import time
from datetime import datetime
import sys
import sqlite3
import threading

pd.options.mode.chained_assignment = None
logging.info('Program Started')
# Get the time to check log
start = time.time()
now = datetime.now()
first_start_time = now.strftime("%H:%M:%S")
print("Start Time =", first_start_time)
sys.stdout.flush()

kp_inventory_file = '../data/kp_inventory/KP_Inventory_file.xlsx'
#kp_inventory_file = '../data/kp_inventory/KP_Inventory_All_giant.xlsx'
# kp_inventory_file = '../data/kp_inventory/KP_Inventory_file_temp.xlsx'
# kp_inventory_file = '../data/kp_inventory/KP_Inventory_All.xlsx'
db_path = "../data/xARA_DB.db"

db = clsSQLiteDatabase(db_path)

# Read excel into a pandas dataframe
df = pd.read_excel(kp_inventory_file, sheet_name=0)

# Get all subjects and objects
unique_subjects = df['subject'].unique()
unique_objects = df['object'].unique()
unique_predicates = df['predicate'].unique()

# Combine unique subjects and objects to form unique nodes
uniqueNodes = np.append(unique_subjects, unique_objects, axis=None)

possible_triplet_list = []
# Get list of all nodes combination
for sub in uniqueNodes:
    for obj in uniqueNodes:
        for pred in unique_predicates:
            possible_triplet_list.append((sub, obj, pred))

possible_triplet_list = list(set(possible_triplet_list))

logging.info('possible triplet list computed')
print('possible_triplet_list len', len(possible_triplet_list))
print(time.perf_counter())
sys.stdout.flush()

# List of tuples of available triplets
available_triplet_list = list(zip(df.subject, df.object, df.predicate))

logging.info('Available triplet list computed')
print('available_triplet_list len', len(available_triplet_list))

sys.stdout.flush()
# List of triplets that need expansions
needs_expansion_list = list(set(possible_triplet_list) - set(available_triplet_list))

print('needs_expansion_list len', len(needs_expansion_list))
print(time.perf_counter())

# List of all the second order expanded quintuplets derived from available nodes list
possible_second_order_expansion_list = []
for atl1 in available_triplet_list:
    for atl2 in available_triplet_list:
        if atl1 != atl2:
            if atl1[1] == atl2[0] and atl1[0] != atl2[1]:
                # n0, n1, n2, e0, e1
                possible_second_order_expansion_list.append((atl1[0], atl1[1], atl2[1], atl1[2], atl2[2]))

possible_second_order_expansion_list = list(set(possible_second_order_expansion_list))
print('possible_second_order_expansion_list len', len(possible_second_order_expansion_list))
# Second order expansion nodes
# Gets the second order expansions to the list of node pairs that needs expansion
# lookup into all_second_order_expansion_list

print("possible_second_order_expansion_list compute Time", time.perf_counter())

sys.stdout.flush()

logging.info('possible_second_order_expansion_list computed')

# Backing up the computed data set. Need to work later to implement bookmarking feature.
with open('possible_second_order_expansion_list.pkl', 'wb') as f:
    pickle.dump(possible_second_order_expansion_list, f)

with open('needs_expansion_list.pkl', 'wb') as f:
    pickle.dump(needs_expansion_list, f)

second_order_expansion_list_stg = []

print('possible_second_order_expansion_list computed ', str(time.perf_counter()))

print('Processing for second order expansions started ', str(time.perf_counter()))

# Get node pairs of psoel and nel
psoel_nodes = [(x[0], x[2]) for x in possible_second_order_expansion_list]
nel_nodes = [(x[0], x[1]) for x in needs_expansion_list]

# Get the nodes that is common in both
common_nodes = set(psoel_nodes).intersection(set(nel_nodes))
# Change it back to list from set
common_nodes = list(common_nodes)

# Converting the list of tuples into pandas dataframe
psoel_df = pd.DataFrame(possible_second_order_expansion_list, columns=['n0', 'n1_1', 'n1', 'e0', 'e1'])
nel_df = pd.DataFrame(needs_expansion_list, columns=['n0', 'n1', 'e0'])

# Read sqlite query results of local similarity values for predicates into a pandas DataFrame
con = sqlite3.connect(db_path)
localSimPred_df = pd.read_sql_query("SELECT * from xARA_LocalSimPredicates", con)
# Close the opened connection
con.close()


def join_and_filter(common_nodes, thread_num):
    """
    function to get the common dataset between psoel and nel
    Computes local similarity between the predicates, ranks them and selects just one of the possible expansions
    """
    print(time.perf_counter())
    staging_df = pd.DataFrame(columns=['n0', 'n1', 'e0', 'n1_1', 'e0_0', 'e1_0'])
    for cn in common_nodes:
        print(str(thread_num), str(cn))  # Thread number and current node pair is printed for logging purposes
        psoel_df_stg = psoel_df[(psoel_df["n0"] == cn[0]) & (psoel_df["n1"] == cn[1])]

        result = pd.merge(nel_df, psoel_df_stg, on=["n0", "n1"])

        # Filter the non required data
        result = result.rename(columns={'e0_x': 'e0', 'e0_y': 'e0_0', 'e1': 'e1_0'})
        sim_score_test_df = pd.merge(result, localSimPred_df, left_on=['e0', 'e0_0'],
                                     right_on=['NEW_CASE_PREDICATE', 'CANDIDATE_CASE_PREDICATE'])

        sim_score_filtered_df = sim_score_test_df[sim_score_test_df['SIMILARITY_SCORE'] >= 0.4]
        sim_score_test_df = pd.merge(sim_score_filtered_df, localSimPred_df, left_on=['e0', 'e1_0'],
                                     right_on=['NEW_CASE_PREDICATE', 'CANDIDATE_CASE_PREDICATE'])

        # Check if sim score test has data in it
        sim_score_filtered_df = sim_score_test_df[sim_score_test_df['SIMILARITY_SCORE_y'] >= 0.4]
        if len(sim_score_filtered_df) > 0:
            sim_score_filtered_df['maxSim'] = sim_score_filtered_df[['SIMILARITY_SCORE_x', 'SIMILARITY_SCORE_y']].apply(
                max, axis=1)
            sim_score_filtered_df["rank"] = sim_score_filtered_df.groupby(["n0", "n1", "e0"])["maxSim"].rank(
                method="first",
                ascending=False)
            sim_score_filtered_df = sim_score_filtered_df[sim_score_filtered_df['rank'] == 1]

            sim_score_filtered_df = sim_score_filtered_df.drop(["NEW_CASE_PREDICATE_x", "CANDIDATE_CASE_PREDICATE_x",
                                                                "SIMILARITY_SCORE_x", "NEW_CASE_PREDICATE_y",
                                                                "CANDIDATE_CASE_PREDICATE_y"
                                                                   , "SIMILARITY_SCORE_y", "maxSim", "rank"
                                                                ], axis=1)

            staging_df = staging_df.append(sim_score_filtered_df)

    # Pickling the dataframe for backup and later usage
    staging_df.to_pickle("staging_df_" + str(thread_num) + ".pkl")
    print(time.perf_counter())


#######################################################################################################################


logging.info("3 threads run the process of filtering and ranking the dataset")

# Preparing for the multithreading, calculating and spreading the number of common node pairs
third = len(common_nodes) // 3
print(third)
second_third = len(common_nodes) * 2 // 3
last = len(common_nodes) - 1

# creating thread
t1 = threading.Thread(target=join_and_filter, args=(common_nodes[0:third], 1,))
t2 = threading.Thread(target=join_and_filter, args=(common_nodes[third + 1:second_third], 2,))
t3 = threading.Thread(target=join_and_filter, args=(common_nodes[second_third + 1:last], 3,))

# starting thread 1
t1.start()
# starting thread 2
t2.start()
# starting thread 3
t3.start()

# wait until thread 1 is completely executed
t1.join()

# wait until thread 2 is completely executed
t2.join()

# wait until thread 3 is completely executed
t3.join()

# three threads completely executed
logging.info("Multi threading part is complete!")

#######################################################################################################################
# Load KP Inventory into database
logging.info('DB Load Started')

engine = create_engine('sqlite:///' + db_path, echo=True)
kp_inventory_stage_table = "KP_INVENTORY_STG"
sqlite_connection = engine.connect()

# Load the data from file into stage table
try:
    df.to_sql(kp_inventory_stage_table, sqlite_connection, if_exists='replace')

except Exception as e:
    print("Error! : ", str(e))

finally:
    sqlite_connection.close()

try:
    # Load from kp inventory stage table into kp inventory table
    kp_inventory_table = 'xARA_KP_Inventory'
    db.start_connection()
    print(db)
    sys.stdout.flush()
    logging.info("Resetting table: " + kp_inventory_table)

    reset_kp_inv_query = "DELETE FROM " + kp_inventory_table
    # sys.stdout.flush()
    db.run_query(reset_kp_inv_query)
    db.run_query("commit;")

    print("Inserting into: ", kp_inventory_table, " from ", kp_inventory_stage_table)

    insert_query = """
                INSERT INTO """ + kp_inventory_table + """
                    (
                    No_of_nodes
                    ,No_of_edges
                    ,N00_category
                    ,N01_category
                    ,N02_category
                    ,N03_category
                    ,E00_predicate
                    ,E00_ends
                    ,E01_predicate
                    ,E01_ends
                    ,E02_predicate
                    ,E02_ends
                    ,KP_to_query
                    )
                select distinct
                    2
                    , 1
                    , subject
                    , object
                    , null
                    , null
                    , predicate
                    , 'n00:n01'
                    , null
                    , null
                    , null
                    , null
                    , kp_name
                from """ + kp_inventory_stage_table + ";"

    db.run_query(insert_query)

    print("Dropping table: ", kp_inventory_stage_table)
    sys.stdout.flush()
    drop_query = "drop table if exists " + kp_inventory_stage_table + ";"
    db.run_query(drop_query)
    db.run_query("commit;")

    #############################################################################
    # Load second order expansion results into database
    logging.info("Start the load of pickled second order dataframes")

    unpickled_df_1 = pd.read_pickle("staging_df_1.pkl")
    unpickled_df_2 = pd.read_pickle("staging_df_2.pkl")
    unpickled_df_3 = pd.read_pickle("staging_df_3.pkl")

    engine = create_engine('sqlite:///' + db_path, echo=True)
    second_order_expansion_stage = "xARA_SecondOrderExpansions_Stage"
    sqlite_connection = engine.connect()

    # Load the data from file into stage table
    try:
        unpickled_df_1.to_sql(second_order_expansion_stage, sqlite_connection, if_exists='replace')
        unpickled_df_2.to_sql(second_order_expansion_stage, sqlite_connection, if_exists='append')
        unpickled_df_3.to_sql(second_order_expansion_stage, sqlite_connection, if_exists='append')

        # db.run_query("commit;")
    except Exception as e:
        print("Error! : ", str(e))

    finally:
        sqlite_connection.close()

    # If expansions have same first and second or same second and third nodes, those need to be removed
    logging.info("Removing expansions with repeated nodes")
    reset_query = """delete from xARA_SecondOrderExpansions
                        where Expanded_N00_category = Expanded_N01_category
                        or Expanded_N01_category = Expanded_N02_category
                        """
    db.run_query(reset_query)

    db.run_query("commit;")

    #############################################################################
    # Load the inventory triplet into case problem table

    print("Resetting table: xARA_CaseProblems")
    reset_query = "DELETE FROM xARA_CaseProblems;"
    db.run_query(reset_query)

    # Loading triplet queries from inventory into xARA_CaseProblems table
    print("Loading inventory triplets into: xARA_CaseProblems")
    query = """
            insert into xARA_CaseProblems
            (
                CASE_ID
                ,NO_OF_EDGES
                ,NO_OF_NODES
                ,N00_NODE_CATEGORY
                ,N01_NODE_CATEGORY
                ,N02_NODE_CATEGORY
                ,N03_NODE_CATEGORY
                ,E00_EDGE_PREDICATE
                ,E00_EDGE_ENDS
                ,E01_EDGE_PREDICATE
                ,E01_EDGE_ENDS
                ,E02_EDGE_PREDICATE
                ,E02_EDGE_ENDS
                ,ORIGIN
            )
            select 
                row_number() over () as case_id
                ,no_of_edges
                ,no_of_nodes
                ,n00_category
                ,n01_category
                ,n02_category
                ,n03_category
                ,e00_predicate
                ,e00_ends
                ,e01_predicate
                ,e01_ends
                ,e02_predicate
                ,e02_ends
                ,'fromKP'
            from
                (SELECT distinct
                    no_of_edges
                    ,no_of_nodes
                    ,n00_category
                    ,n01_category
                    ,n02_category
                    ,n03_category
                    ,e00_predicate
                    ,e00_ends
                    ,e01_predicate
                    ,e01_ends
                    ,e02_predicate
                    ,e02_ends
                FROM 
                    xARA_KP_Inventory)
            """ 
    db.run_query(query)
    db.run_query('commit;')

    # Loading triplet queries from second order expansions into xARA_CaseProblems table
    print("Loading triplet queries from second order expansions into: xARA_CaseProblems")
    query = """
                insert into xARA_CaseProblems
                (
                    CASE_ID
                    ,NO_OF_EDGES
                    ,NO_OF_NODES
                    ,N00_NODE_CATEGORY
                    ,N01_NODE_CATEGORY
                    ,N02_NODE_CATEGORY
                    ,N03_NODE_CATEGORY
                    ,E00_EDGE_PREDICATE
                    ,E00_EDGE_ENDS
                    ,E01_EDGE_PREDICATE
                    ,E01_EDGE_ENDS
                    ,E02_EDGE_PREDICATE
                    ,E02_EDGE_ENDS
                    ,ORIGIN
                )
                select
                    row_number() over () + (select max(cast(case_id as integer)) from xARA_CaseProblems) as case_id
                    ,1 no_of_edges
                    ,2 no_of_nodes
                    ,n0 n00_category
                    ,n1 n01_category
                    ,null n02_category
                    ,null n03_category
                    ,e0 e00_predicate
                    ,'n00:n01' e00_ends
                    ,null e01_predicate
                    ,null e01_ends
                    ,null e02_predicate
                    ,null e02_ends 
                    ,'derived'
                    from xARA_SecondOrderExpansions_Stage
                """
    db.run_query(query)
    db.run_query('commit;')

    # Updating case ID to be in the format Q000000
    update_query = "update xARA_CaseProblems set case_id = 'Q' || substr(replace(hex(zeroblob(6)),'00','0'),1," \
                   "6 - length(CASE_ID))|| CASE_ID; "
    db.run_query(update_query)
    db.run_query('commit;')

    #############################################################################
    # Load the inventory triplet into case problem table

    print("Resetting table: xARA_CaseSolutions")
    reset_query = "DELETE FROM xARA_CaseSolutions;"
    db.run_query(reset_query)

    # Loading triplet query solution from inventory into xARA_CaseSolutions table
    print('Loading triplet query solution from inventory into xARA_CaseSolutions table')
    query = """
                insert into xARA_CaseSolutions
                (
                    CASE_ID
                    ,ORIGIN
                    ,SOLUTION_STEPS
                    ,SOLUTION_FIRST_KP_NAME
                    ,SOLUTION_SECOND_KP_NAME
                    ,NODE1_PATH1_CATEGORY
                    ,NODE2_PATH1_CATEGORY
                    ,NODE1_PATH2_CATEGORY
                    ,NODE2_PATH2_CATEGORY
                    ,EDGE1_PATH1_PREDICATE
                    ,EDGE1_PATH1_ENDS
                    ,EDGE1_PATH2_PREDICATE
                    ,EDGE1_PATH2_ENDS
                )
                select 	
                    cp.CASE_ID
                    ,cp.ORIGIN
                    ,1 SOLUTION_STEPS
                    ,inv.kp_to_query SOLUTION_FIRST_KP_NAME
                    ,null SOLUTION_SECOND_KP_NAME
                    ,cp.n00_node_category NODE1_PATH1_CATEGORY
                    ,cp.n01_node_category NODE2_PATH1_CATEGORY
                    ,null NODE1_PATH2_CATEGORY
                    ,null NODE2_PATH2_CATEGORY
                    ,cp.e00_edge_predicate EDGE1_PATH1_PREDICATE
                    ,cp.e00_edge_ends EDGE1_PATH1_ENDS
                    ,null EDGE1_PATH2_PREDICATE
                    ,null EDGE1_PATH2_ENDS
                from xARA_CaseProblems cp
                inner join xARA_KP_Inventory inv
                    on inv.N00_category = cp.N00_NODE_CATEGORY
                    and inv.n01_category = cp.N01_NODE_CATEGORY
                    and inv.e00_predicate = cp.E00_EDGE_PREDICATE
                where cp.origin = 'fromKP'

                """
    db.run_query(query)
    db.run_query('commit')

    ###########################################################################################
    print("Loading query solution from second order expansions into xARA_CaseSolutions table")
    query = """insert into xARA_CaseSolutions
                (
                    CASE_ID
                    ,ORIGIN
                    ,SOLUTION_STEPS
                    ,SOLUTION_FIRST_KP_NAME
                    ,SOLUTION_SECOND_KP_NAME
                    ,NODE1_PATH1_CATEGORY
                    ,NODE2_PATH1_CATEGORY
                    ,NODE1_PATH2_CATEGORY
                    ,NODE2_PATH2_CATEGORY
                    ,EDGE1_PATH1_PREDICATE
                    ,EDGE1_PATH1_ENDS
                    ,EDGE1_PATH2_PREDICATE
                    ,EDGE1_PATH2_ENDS
                )
            with soe_solution as(
            select
                n0 n00_category
                ,n1 n01_category
                ,e0 e00_predicate
                ,n0 Expanded_N00_category
                ,n1_1 Expanded_N01_category
                ,n1 Expanded_N02_category
                ,e0_0 Expanded_E00_predicate
                ,e1_0 Expanded_E01_predicate
            from xARA_SecondOrderExpansions_Stage
            )
            select cp.case_id
                , cp.origin
                , 2 solution_steps
                , kp1.kp_to_query solution_first_kp_name
                , kp2.kp_to_query solution_second_kp_name
                , soe.Expanded_N00_category node1_path1_category
                , soe.Expanded_N01_category node2_path1_category
                , soe.Expanded_N01_category node1_path2_category
                , soe.Expanded_N02_category node2_path2_category
                , soe.Expanded_E00_predicate EDGE1_PATH1_PREDICATE
                ,'n00:n01' EDGE1_PATH1_ENDS
                , soe.Expanded_E01_predicate EDGE1_PATH2_PREDICATE
                ,'n01:n02' EDGE1_PATH2_ENDS
            from soe_solution soe
            inner join xARA_CaseProblems cp on cp.N00_NODE_CATEGORY = soe.N00_category
                                            and cp.N01_NODE_CATEGORY = soe.N01_category
                                            and cp.E00_EDGE_PREDICATE = soe.E00_predicate
            inner join xARA_KP_Inventory kp1 on soe.Expanded_N00_category = kp1.N00_category
                                            and soe.Expanded_N01_category = kp1.N01_category
                                            and soe.Expanded_E00_predicate = kp1.E00_predicate
            inner join xARA_KP_Inventory kp2 on soe.Expanded_N01_category = kp2.N00_category
                                            and soe.Expanded_N02_category = kp2.N01_category
                                            and soe.Expanded_E01_predicate = kp2.E00_predicate
            ;
            """
    db.run_query(query)
    db.run_query('commit')

    # Update the node and edge information in the database

    """ delete from xARA_Predicates;

        delete from xARA_Nodes;

        insert into xARA_Nodes
        (
        node_category
        )
        select distinct N00_category from xARA_KP_Inventory
        UNION
        select distinct n01_category from xARA_KP_Inventory
        ;


        insert into xARA_Predicates
        (
        predicate_name
        )
        select distinct E00_predicate from xARA_KP_Inventory;"""

    logging.info('DB Load Ended')

except Exception as e:
    print("Error! : ", str(e))
    raise e

finally:
    db.end_connection()

    end = time.time()
    # total time taken
    print(f"Runtime of the program is {end - start} seconds")
