import plotly.graph_objects as go
import numpy as np
import random
from danpy.sb import dsb
import pandas as pd
import matplotlib
import pickle
from scipy.optimize import fsolve


l_Ts_over_l_To = 0.95
l_Ts_over_l_To_min = 1/1.07
l_Ts_over_l_To_max = 1/1.03
initial_force = 0.001
kT_intersection = (
    (l_Ts_over_l_To - 1)
    / (
        np.log(
            np.exp(initial_force/0.20)
            - 1
        )
        - 1/0.20
    )
)
max_cT = 100

#########################################
###### CALCULATE CONSTRAINED CT/KT ######
#########################################

# kT_array = np.linspace(0,kT_intersection,1001)
#
# previous_params = pickle.load(open( "previous_params.pkl", "rb" ) )
# if (
#         (previous_params["l_Ts_over_l_To"]==l_Ts_over_l_To)
#         and (previous_params["initial_force"]==initial_force)
#         ):
#     lb_cT = previous_params["lb_cT"]
#     ub_cT = previous_params["ub_cT"]
# else:
#     long_cT_array = np.linspace(10,100,10001)
#     lb_cT = np.zeros(np.shape(kT_array))
#     ub_cT = np.zeros(np.shape(kT_array))
#     for i in range(len(kT_array)):
#         if (long_cT_array*kT_array[i]<0.20).all():
#             ub_cT[i] = 100
#         else:
#             ub_cT[i] = 0.20/kT_array[i]
#         if kT_array[i]==0:
#             lb_cT[i] = (1-initial_force)/(1-l_Ts_over_l_To)
#         else:
#             lT_initial = (1/l_Ts_over_l_To)*(
#                 kT_array[i]*np.log(
#                     np.exp(initial_force/(long_cT_array*kT_array[i]))
#                     - 1
#                 )
#                 + (1-1/long_cT_array)
#             )
#             lb_cT[i] = long_cT_array[int(sum(lT_initial<1))-1]
#
# new_params = {
#     "l_Ts_over_l_To" : l_Ts_over_l_To,
#     "initial_force" : initial_force,
#     "lb_cT" : lb_cT,
#     "ub_cT" : ub_cT
# }
# pickle.dump(new_params, open("previous_params.pkl", "wb"))

####################################
###### DEFINE PARAMETER LISTS ######
####################################

CT = []

KT = []

Pennation_min = 0*np.pi/180
Pennation_max = 40*np.pi/180
Pennation = []

Slack_Opt_Ratio_min = 0.1
Slack_Opt_Ratio_max = 11.25
Slack_Opt_Ratio = []

Slack_Opt_Tendon_Ratio_min = l_Ts_over_l_To_min
Slack_Opt_Tendon_Ratio_max = l_Ts_over_l_To_max
Slack_Opt_Tendon_Ratio = []

MVC = []

Error = []

################################
###### FIND RANDOM POINTS ######
################################

Seed = None
np.random.seed(Seed)
random.seed(Seed)
numTrials = 10000
count = 0
breakcount = 0
statusbar = dsb(0,numTrials,title="Parallel Coordinates")
while count<numTrials:
    l_Ts_over_l_To_rand = random.uniform(l_Ts_over_l_To_min,l_Ts_over_l_To_max)
    # l_Ts_over_l_To_rand = l_Ts_over_l_To
    kT_intersection = (
        (l_Ts_over_l_To_rand - 1)
        / (
            np.log(
                np.exp(initial_force/0.20)
                - 1
            )
            - 1/0.20
        )
    )
    kT_rand = random.uniform(0.0001,kT_intersection)
    try:
        # lb_cT = fsolve(
        #     lambda cT: (
        #         (1/l_Ts_over_l_To_min)*(
        #             kT_rand*np.log(
        #                 np.exp(initial_force/(cT*kT_rand))
        #                 - 1
        #             )
        #             + (1-1/cT)
        #         )
        #         - 1
        #     ),
        #     20
        # )[0]
        # ub_cT = fsolve(
        #     lambda cT: (
        #         (1/l_Ts_over_l_To_max)*(
        #             kT_rand*np.log(
        #                 np.exp(initial_force/(cT*kT_rand))
        #                 - 1
        #             )
        #             + (1-1/cT)
        #         )
        #         - 1
        #     ),
        #     20
        # )[0]
        # if kT_rand*ub_cT>0.20:
        #     ub_cT=0.20/kT_rand
        cT_rand = fsolve(
            lambda cT: (
                (1/l_Ts_over_l_To_rand)*(
                    kT_rand*np.log(
                        np.exp(initial_force/(cT*kT_rand))
                        - 1
                    )
                    + (1-1/cT)
                )
                - 1
            ),
            20
        )[0]
        assert cT_rand*kT_rand<=0.20, "Doesn't meet upper bound condition."
    except:
        pass

    slack_opt_ratio_rand = np.random.uniform(
        Slack_Opt_Ratio_min,
        Slack_Opt_Ratio_max
    )
    pennation_rand = np.random.uniform(Pennation_min,Pennation_max)
    # pennation_rand = 0
    mvc_rand = np.random.uniform(
        initial_force/np.cos(pennation_rand),
        1
    )
    error = (
        -slack_opt_ratio_rand
        * (kT_rand/np.cos(pennation_rand))
        * (1/l_Ts_over_l_To_rand)
        * np.log(
            (np.exp((mvc_rand*np.cos(pennation_rand))/(cT_rand*kT_rand)) - 1)
            / (np.exp(initial_force/(cT_rand*kT_rand)) - 1)
        )
    )
    if error>=-1:
        CT.append(cT_rand)
        KT.append(kT_rand)
        MVC.append(mvc_rand)
        Slack_Opt_Tendon_Ratio.append(l_Ts_over_l_To_rand)
        Pennation.append(pennation_rand)
        Slack_Opt_Ratio.append(slack_opt_ratio_rand)
        Error.append(error)
        statusbar.update(count)
        count+=1
    else:
        breakcount+=1
    if breakcount>=1000:
        import ipdb; ipdb.set_trace()

############################################
###### ERROR DICT FOR PARALLEL COORDS ######
############################################

if min(Error)>-np.floor((1/l_Ts_over_l_To - 1)*Slack_Opt_Ratio_max/0.1)*0.1:
    errorTickVals = list(
        np.arange(
            -100*np.floor((1/l_Ts_over_l_To - 1)*Slack_Opt_Ratio_max/0.1)*0.1,
            1e-7,
            100*0.1
        )
    )
    errorTickLabels = ["{0:.2f}".format(el) for el in errorTickVals]
    errorTickLabels[-1]="0"
    errorConstraintRange = [
        -100*np.floor((1/l_Ts_over_l_To - 1)*Slack_Opt_Ratio_max/0.1)*0.1 + 20,
        -100*np.floor((1/l_Ts_over_l_To - 1)*Slack_Opt_Ratio_max/0.1)*0.1
    ]
else:
    errorTickVals = list(
        np.concatenate(
            [
                [100*min(Error)],
                np.arange(
                    -100*np.floor((1/l_Ts_over_l_To - 1)*Slack_Opt_Ratio_max/0.1)*0.1,
                    1e-7,
                    100*0.1
                )
            ]
        )
    )
    errorTickLabels = ["{0:.2f}".format(el) for el in errorTickVals]
    errorTickLabels[0]=""
    errorTickLabels[-1]="0"
    errorConstraintRange = [
        100*min(Error)+20,
        100*min(Error)
    ]

Error_dict = dict(
    # constraintrange = [min(Error)*100,-30],
    range = [
        100*min(Error),
        0
    ],
    # tickvals = errorTickVals,
    # ticktext = errorTickLabels,
    label = 'Percent Fascicle Length Error',
    values = list(100*np.array(Error)),
)

################################################
###### CURVATURE DICT FOR PARALLEL COORDS ######
################################################

curvatureTickVals = list(
    np.arange(
        0,
        np.ceil(kT_intersection/0.001)*0.001 + 1e-7,
        0.001
    )
)
curvatureTickLabels = ["{0:.3f}".format(el) for el in curvatureTickVals]
curvatureTickLabels[0] = "0"
# curvatureTickLabels[-1] = ""

Curvature_dict = dict(
    range = [
        0,
        max(KT)
    ],
    # tickvals = curvatureTickVals,
    # ticktext = curvatureTickLabels,
    label = 'Tendon Radius of Curvature Constant',
    values = KT
)

################################################
###### STIFFNESS DICT FOR PARALLEL COORDS ######
################################################

Stiffness_dict = dict(
    # range = [20,100],
    # tickvals = [20,30,40,50,60,70,80,90,100],
    label = 'Norm. Asymptotic Tendon Stiffness',
    values = CT
)

##########################################################
### SLACK/OPTIMAL TENDON RATIO DICT FOR PARALLEL COORDS ##
##########################################################

Tendon_Ratio_dict = dict(
    range = [
        1/Slack_Opt_Tendon_Ratio_max,
        1/Slack_Opt_Tendon_Ratio_min
    ],
    # tickvals = RatioTickVals,
    label = 'Optimal Tendon Length/Tendon Slack Length',
    values = [1/el for el in Slack_Opt_Tendon_Ratio]
)

##########################################################
###### SLACK/OPTIMAL RATIO DICT FOR PARALLEL COORDS ######
##########################################################

RatioTickVals = list(
    np.arange(
        0,
        Slack_Opt_Ratio_max +1e-3,
        2
    )
)
RatioTickVals.append(Slack_Opt_Ratio_max)

Ratio_dict = dict(
    range = [
        0,
        Slack_Opt_Ratio_max
    ],
    # tickvals = RatioTickVals,
    label = 'Tendon Slack Length/Optimal Fascicle Length',
    values = Slack_Opt_Ratio
)

################################################
###### PENNATION DICT FOR PARALLEL COORDS ######
################################################

Pennation_dict = dict(
    range = [
        180*Pennation_min/np.pi,
        180*Pennation_max/np.pi
    ],
    # tickvals = list(
    #     np.arange(
    #         180*Pennation_min/np.pi,
    #         180*Pennation_max/np.pi + 1e-3,
    #         5
    #     )
    # ),
    label = 'Pennation Angle (deg.)',
    values = list(180*np.array(Pennation)/np.pi)
)

##########################################
###### MVC DICT FOR PARALLEL COORDS ######
##########################################

MVC_dict = dict(
    range = [
        0,
        np.ceil(100*max(MVC))
    ],
    label = 'Percent MVC',
    values = list(100*np.array(MVC))
    # constraintrange = [95,100]
    # tickvals = list(np.linspace(0,100,11))
)


####################################
###### PARALLEL COORDS FIGURE ######
####################################

# Left out Pennation_dict,
fig = go.Figure(data=
    go.Parcoords(
        line = dict(
            color = pd.Series([-el for el in Error],name='Error'),
            colorscale = 'plasma'
        ),
        dimensions = list([
            Error_dict,
            Stiffness_dict,
            Curvature_dict,
            Tendon_Ratio_dict,
            Ratio_dict,
            Pennation_dict,
            MVC_dict
        ])
    )
)
# fig.show()
# fig.write_image("Figures/Iso_Error_PC/v3/PC_all.pdf")
fig.write_html('tendon_length_change_parallel_coords_2.html', auto_open=True)
