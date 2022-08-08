# Explanatory-Agent-DB

The following contains the scripts used to generate different tables in xARA_DB by Explanatory Agent as part of the Translator (NIH) project.

The scripts and their corresponding contributions are as follows:

1. create_kp_inventory.py
This script is used to create the KP Inventory table within xARA_DB. The above pulls data from xARA_KP_Info and creates an Excel file (KP_Inventory_File) with all nodes and edges across all Translator KPs.

2. load_local_similarity_nodes_predicates.py
Obtains all relations from the KP_Inventory_File and calculates the similarity score between different node categories and edge predicates based on the Biolink Model Toolkit (BMT) tree (xARA_LocalSimNodes & xARA_LocalSimPredicates).


3. load_casebase_xARA.py
Creates the xARA_Caseproblems and xARA_CaseSolutions using the KP_Inventory_File created using Script 1. This script builds Case Problems and Case Solutions based on the list of triplets fron the KP_Inventory_file.
