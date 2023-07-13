from sole import Sole
from solepair import SolePair
from solepaircompare import SolePairCompare
from util import METADATA


def process_image(Q_file, K_file, mated, partial_type="full"):
    Q = Sole("2DScanImages/" + Q_file, border_width=160)
    K = Sole("2DScanImages/" + K_file, border_width=160)
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

    if partial_type == "full":
        sc = SolePairCompare(pair, icp_downsample_rate=0.2, two_way=True, shift_left=True,
                             shift_right=True, shift_down=True, shift_up=True)  # icp is called here
    elif partial_type == "toe":
        toe_coords = Q.coords[Q.coords.x <= middle_x]
        toe = Sole(image_path="2DScanImages/"+Q_file,
                   is_image=False, coords=toe_coords)
        pair = SolePair(toe, K, mated=mated)
        sc = SolePairCompare(pair, icp_downsample_rate=0.2,
                             two_way=True, shift_right=True)
    elif partial_type == "heel":
        heel_coords = Q.coords[Q.coords.x > middle_x]
        heel = Sole(image_path="2DScanImages/"+Q_file,
                    is_image=False, coords=heel_coords)
        pair = SolePair(heel, K, mated=mated)
        sc = SolePairCompare(pair, icp_downsample_rate=0.2,
                             two_way=True, shift_left=True)
    elif partial_type == "inside":
        if q_foot == "R":
            inside_coords = Q.coords[Q.coords.y <= middle_y]
            inside = Sole(image_path="2DScanImages/"+Q_file,
                          is_image=False, coords=inside_coords)
            pair = SolePair(inside, K, mated=mated)
            sc = SolePairCompare(
                pair, icp_downsample_rate=0.2, two_way=True, shift_up=True)
        else:
            inside_coords = Q.coords[Q.coords.y >= middle_y]
            inside = Sole(image_path="2DScanImages/"+Q_file,
                          is_image=False, coords=inside_coords)
            pair = SolePair(inside, K, mated=mated)
            sc = SolePairCompare(pair, icp_downsample_rate=0.2,
                                 two_way=True, shift_down=True)
    elif partial_type == "outside":
        if q_foot == "R":
            outside_coords = Q.coords[Q.coords.y > middle_y]
            outside = Sole(image_path="2DScanImages/"+Q_file,
                           is_image=False, coords=outside_coords)
            pair = SolePair(outside, K, mated=mated)
            sc = SolePairCompare(pair, icp_downsample_rate=0.2,
                                 two_way=True, shift_down=True)
        else:
            outside_coords = Q.coords[Q.coords.y < middle_y]
            outside = Sole(image_path="2DScanImages/"+Q_file,
                           is_image=False, coords=outside_coords)
            pair = SolePair(outside, K, mated=mated)
            sc = SolePairCompare(
                pair, icp_downsample_rate=0.2, two_way=True, shift_up=True)

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
    row.update(sc.cluster_metrics())
    row['q_pct_threshold_1'] = sc.percent_overlap(Q_as_base=True, threshold=1)
    row['k_pct_threshold_1'] = sc.percent_overlap(Q_as_base=False, threshold=1)

    row['q_pct_threshold_2'] = sc.percent_overlap(Q_as_base=True, threshold=2)
    row['k_pct_threshold_2'] = sc.percent_overlap(Q_as_base=False, threshold=2)

    row['q_pct_threshold_3'] = sc.percent_overlap(Q_as_base=True, threshold=3)
    row['k_pct_threshold_3'] = sc.percent_overlap(Q_as_base=False, threshold=3)

    row['q_pct_threshold_5'] = sc.percent_overlap(Q_as_base=True, threshold=5)
    row['k_pct_threshold_5'] = sc.percent_overlap(Q_as_base=False, threshold=5)

    row['q_pct_threshold_10'] = sc.percent_overlap(Q_as_base=True, threshold=10)
    row['k_pct_threshold_10'] = sc.percent_overlap(Q_as_base=False, threshold=10)

    return row
