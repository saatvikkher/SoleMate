from sole import Sole
from solepair import SolePair
from solepaircompare import SolePairCompare
from util import METADATA, METADATA_BLURRY, METADATA_OOD, OPP_FOOT


def process_image(Q_file, K_file, mated, partial_type="full", folder_path="2DScanImages/"):
    Q = Sole(folder_path + Q_file, border_width=160)
    K = Sole(folder_path + K_file, border_width=160)
    pair = SolePair(Q, K, mated=mated)

    q_row = METADATA[METADATA['File Name'] == Q_file]

    q_file_name = q_row['File Name'].iloc[0]
    q_number = q_row['Shoe Number'].iloc[0]
    q_model = q_row['Shoe Make/Model'].iloc[0]
    q_size = q_row['Shoe Size'].iloc[0]
    q_foot = Q_file[6:7]

    k_row = METADATA[METADATA['File Name'] == K_file]

    k_file_name = k_row['File Name'].iloc[0]
    k_number = k_row['Shoe Number'].iloc[0]
    k_model = k_row['Shoe Make/Model'].iloc[0]
    k_size = k_row['Shoe Size'].iloc[0]
    k_foot = K_file[6:7]

    x_series = Q.coords.x
    y_series = Q.coords.y

    middle_x = x_series.quantile(0.025) + ((x_series.quantile(0.975) - x_series.quantile(0.025)) / 2)
    middle_y = y_series.quantile(0.025) + ((y_series.quantile(0.975) - y_series.quantile(0.025)) / 2)

    if partial_type == "toe":
        toe_coords = Q.coords[Q.coords.x <= middle_x]
        toe = Sole(image_path=folder_path+Q_file,
                   is_image=False, coords=toe_coords)
        pair = SolePair(toe, K, mated=mated)
        
    elif partial_type == "heel":
        heel_coords = Q.coords[Q.coords.x > middle_x]
        heel = Sole(image_path=folder_path+Q_file,
                    is_image=False, coords=heel_coords)
        pair = SolePair(heel, K, mated=mated)
        
    elif partial_type == "inside":
        if q_foot == "R":
            inside_coords = Q.coords[Q.coords.y <= middle_y]
        else:
            inside_coords = Q.coords[Q.coords.y >= middle_y]
        inside = Sole(image_path=folder_path+Q_file,
                        is_image=False, coords=inside_coords)
        pair = SolePair(inside, K, mated=mated)
        
    elif partial_type == "outside":
        if q_foot == "R":
            outside_coords = Q.coords[Q.coords.y > middle_y]
        else:
            outside_coords = Q.coords[Q.coords.y < middle_y]
        outside = Sole(image_path=folder_path+Q_file,
                        is_image=False, coords=outside_coords)
        pair = SolePair(outside, K, mated=mated)
    
    sc = SolePairCompare(pair, 
                         icp_downsample_rates=[0.05, 0.2, 0.5], 
                         two_way=True, 
                         shift_left=True,
                         shift_right=True, 
                         shift_down=True, 
                         shift_up=True)

    # Cut K before running any metric
    sc.cut_k(partial_type, k_foot)

    row = {'q_file_name': q_file_name,
           'k_file_name': k_file_name,
           'q_number': q_number,
           'k_number': k_number,
           'mated': pair.mated,
           'q_model': q_model,
           'k_model': k_model,
           'q_size': q_size,
           'k_size': k_size,
           'q_foot': q_foot,
           'k_foot': k_foot,
           'partial_type': partial_type,
           'q_points_count': len(sc.Q_coords),
           'k_points_count': len(sc.K_coords)}

    # Running metrics
    row.update(sc.min_dist())

    row.update(sc.cluster_metrics(n_clusters = 20))
    row.update(sc.cluster_metrics(n_clusters = 100))
    row.update(sc.cluster_metrics(n_clusters = 500))

    row['q_pct_threshold_1'] = sc.percent_overlap(Q_as_base=True, threshold=1)
    row['k_pct_threshold_1'] = sc.percent_overlap(Q_as_base=False, threshold=1)

    row['q_pct_threshold_2'] = sc.percent_overlap(Q_as_base=True, threshold=2)
    row['k_pct_threshold_2'] = sc.percent_overlap(Q_as_base=False, threshold=2)

    row['q_pct_threshold_3'] = sc.percent_overlap(Q_as_base=True, threshold=3)
    row['k_pct_threshold_3'] = sc.percent_overlap(Q_as_base=False, threshold=3)

    row.update(sc.pc_metrics())
    row.update(sc.jaccard_index())

    return row

def process_image_OOD(Q_file, K_file, mated, partial_type="full", folder_path="OOD/"):
    if folder_path == "OOD/" and mated == False:
        Q = Sole(folder_path + Q_file, border_width=160, flipped=True)
        q_foot = OPP_FOOT[Q_file[7:8]]
    else:
        Q = Sole(folder_path + Q_file, border_width=160)
        q_foot = Q_file[7:8]
    K = Sole(folder_path + K_file, border_width=160)
    pair = SolePair(Q, K, mated=mated)

    q_row = METADATA_OOD[METADATA_OOD['File Name'] == Q_file]

    q_file_name = q_row['File Name'].iloc[0]
    q_number = q_row['ID'].iloc[0]
    q_model = None
    q_size = q_row['Size'].iloc[0]

    k_row = METADATA_OOD[METADATA_OOD['File Name'] == K_file]

    k_file_name = k_row['File Name'].iloc[0]
    k_number = k_row['ID'].iloc[0]
    k_model = None
    k_size = k_row['Size'].iloc[0]
    k_foot = k_row['Foot'].iloc[0]

    x_series = Q.coords.x
    y_series = Q.coords.y

    middle_x = x_series.quantile(0.025) + ((x_series.quantile(0.975) - x_series.quantile(0.025)) / 2)
    middle_y = y_series.quantile(0.025) + ((y_series.quantile(0.975) - y_series.quantile(0.025)) / 2)

    if partial_type == "toe":
        toe_coords = Q.coords[Q.coords.x <= middle_x]
        toe = Sole(image_path=folder_path+Q_file,
                   is_image=False, coords=toe_coords)
        pair = SolePair(toe, K, mated=mated)
    elif partial_type == "heel":
        heel_coords = Q.coords[Q.coords.x > middle_x]
        heel = Sole(image_path=folder_path+Q_file,
                    is_image=False, coords=heel_coords)
        pair = SolePair(heel, K, mated=mated)
    elif partial_type == "inside":
        if q_foot == "R":
            inside_coords = Q.coords[Q.coords.y <= middle_y]
        else:
            inside_coords = Q.coords[Q.coords.y >= middle_y]
        inside = Sole(image_path=folder_path+Q_file,
                        is_image=False, coords=inside_coords)
        pair = SolePair(inside, K, mated=mated)
    elif partial_type == "outside":
        if q_foot == "R":
            outside_coords = Q.coords[Q.coords.y > middle_y]
        else:
            outside_coords = Q.coords[Q.coords.y < middle_y]
        outside = Sole(image_path=folder_path+Q_file,
                        is_image=False, coords=outside_coords)
        pair = SolePair(outside, K, mated=mated)
    
    sc = SolePairCompare(pair, 
                         icp_downsample_rates=[0.05, 0.2, 0.5], 
                         two_way=True, 
                         shift_left=True,
                         shift_right=True, 
                         shift_down=True, 
                         shift_up=True)  # icp is called here

    # Cut K before running any metric
    sc.cut_k(partial_type, k_foot)

    row = {'q_file_name': q_file_name,
           'k_file_name': k_file_name,
           'q_number': q_number,
           'k_number': k_number,
           'mated': pair.mated,
           'q_model': q_model,
           'k_model': k_model,
           'q_size': q_size,
           'k_size': k_size,
           'q_foot': q_foot,
           'k_foot': k_foot,
           'partial_type': partial_type,
           'q_points_count': len(sc.Q_coords),
           'k_points_count': len(sc.K_coords)}

    # Running metrics
    row.update(sc.min_dist())

    row.update(sc.cluster_metrics(n_clusters = 20))
    row.update(sc.cluster_metrics(n_clusters = 100))
    row.update(sc.cluster_metrics(n_clusters = 500))

    row['q_pct_threshold_1'] = sc.percent_overlap(Q_as_base=True, threshold=1)
    row['k_pct_threshold_1'] = sc.percent_overlap(Q_as_base=False, threshold=1)

    row['q_pct_threshold_2'] = sc.percent_overlap(Q_as_base=True, threshold=2)
    row['k_pct_threshold_2'] = sc.percent_overlap(Q_as_base=False, threshold=2)

    row['q_pct_threshold_3'] = sc.percent_overlap(Q_as_base=True, threshold=3)
    row['k_pct_threshold_3'] = sc.percent_overlap(Q_as_base=False, threshold=3)

    row.update(sc.pc_metrics())

    return row

def process_image_blurry(Q_file, K_file, mated, partial_type="full", folder_path="Blurry/"):
    Q = Sole(folder_path + Q_file, border_width=160)
    K = Sole(folder_path + K_file, border_width=160)
    pair = SolePair(Q, K, mated=mated)

    q_row = METADATA_BLURRY[METADATA_BLURRY['File_Name'] == Q_file]

    q_file_name = q_row['File_Name'].iloc[0]
    q_number = q_row['ShoeID'].iloc[0]
    q_model = None
    q_size = q_row['Size'].iloc[0]
    q_foot = q_row['Foot'].iloc[0]

    k_row = METADATA_BLURRY[METADATA_BLURRY['File_Name'] == K_file]

    k_file_name = k_row['File_Name'].iloc[0]
    k_number = k_row['ShoeID'].iloc[0]
    k_model = None
    k_size = k_row['Size'].iloc[0]
    k_foot = k_row['Foot'].iloc[0]

    x_series = Q.coords.x
    y_series = Q.coords.y

    middle_x = x_series.quantile(0.025) + ((x_series.quantile(0.975) - x_series.quantile(0.025)) / 2)
    middle_y = y_series.quantile(0.025) + ((y_series.quantile(0.975) - y_series.quantile(0.025)) / 2)

    if partial_type == "toe":
        toe_coords = Q.coords[Q.coords.x <= middle_x]
        toe = Sole(image_path=folder_path+Q_file,
                   is_image=False, coords=toe_coords)
        pair = SolePair(toe, K, mated=mated)
    
    elif partial_type == "heel":
        heel_coords = Q.coords[Q.coords.x > middle_x]
        heel = Sole(image_path=folder_path+Q_file,
                    is_image=False, coords=heel_coords)
        pair = SolePair(heel, K, mated=mated)
        
    elif partial_type == "inside":
        if q_foot == "R":
            inside_coords = Q.coords[Q.coords.y <= middle_y]
        else:
            inside_coords = Q.coords[Q.coords.y >= middle_y]
        inside = Sole(image_path=folder_path+Q_file,
                        is_image=False, coords=inside_coords)
        pair = SolePair(inside, K, mated=mated)
            
    elif partial_type == "outside":
        if q_foot == "R":
            outside_coords = Q.coords[Q.coords.y > middle_y]
        else:
            outside_coords = Q.coords[Q.coords.y < middle_y]
        outside = Sole(image_path=folder_path+Q_file,
                        is_image=False, coords=outside_coords)
        pair = SolePair(outside, K, mated=mated)
    
    sc = SolePairCompare(pair, 
                         icp_downsample_rates=[0.05, 0.2, 0.5], 
                         two_way=True, 
                         shift_left=True,
                         shift_right=True, 
                         shift_down=True, 
                         shift_up=True)  # icp is called here

    # Cut K before running any metric
    sc.cut_k(partial_type, k_foot)

    row = {'q_file_name': q_file_name,
           'k_file_name': k_file_name,
           'q_number': q_number,
           'k_number': k_number,
           'mated': pair.mated,
           'q_model': q_model,
           'k_model': k_model,
           'q_size': q_size,
           'k_size': k_size,
           'q_foot': q_foot,
           'k_foot': k_foot,
           'partial_type': partial_type,
           'q_points_count': len(sc.Q_coords),
           'k_points_count': len(sc.K_coords)}

    # Running metrics
    row.update(sc.min_dist())

    row.update(sc.cluster_metrics(n_clusters = 20))
    row.update(sc.cluster_metrics(n_clusters = 100))
    row.update(sc.cluster_metrics(n_clusters = 500))

    row['q_pct_threshold_1'] = sc.percent_overlap(Q_as_base=True, threshold=1)
    row['k_pct_threshold_1'] = sc.percent_overlap(Q_as_base=False, threshold=1)

    row['q_pct_threshold_2'] = sc.percent_overlap(Q_as_base=True, threshold=2)
    row['k_pct_threshold_2'] = sc.percent_overlap(Q_as_base=False, threshold=2)

    row['q_pct_threshold_3'] = sc.percent_overlap(Q_as_base=True, threshold=3)
    row['k_pct_threshold_3'] = sc.percent_overlap(Q_as_base=False, threshold=3)

    row.update(sc.pc_metrics())

    return row