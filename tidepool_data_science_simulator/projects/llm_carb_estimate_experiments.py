import sys
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../..')

import copy
import os
import re
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tidepool_data_science_simulator.models.sensor import NoisySensor, IdealSensor
from tidepool_data_science_simulator.models.patient import VirtualPatientModel
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.controller import LoopController, DoNothingController
from tidepool_data_science_simulator.evaluation.inspect_results import load_results
from tidepool_data_science_simulator.makedata.make_simulation import get_canonical_simulation
from tidepool_data_science_simulator.makedata.make_patient import (
    get_canonical_risk_pump_config,
    get_canonical_virtual_patient_model_config,
    get_canonical_sensor_config,
    DATETIME_DEFAULT
)

from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_simulator.run import run_simulations

from tidepool_data_science_simulator.models.events import CarbTimeline, BolusTimeline
from tidepool_data_science_simulator.models.measures import Carb, Bolus

import datetime
from time import time

import matplotlib.dates as mdates
formatter = mdates.DateFormatter('%H:%M')
cmap = plt.cm.plasma_r


def build_metabolic_sensitivity_sims(start_glucose_value=110, basal_rate=0.3, cir=20.0, isf=150.0, target_range_min=100, target_range_max=120, carb_timeline_list=[], pump_carb_timeline_list=[] ,bolus_timeline=None, duration_hrs=1, controller=LoopController):
    """
    Look at resulting bgs from settings that are correct/incorrect for analysis.

    Parameters
    ----------
    scenario_csv_filepath: str
        Path to the scenario file

    param_grid: list of dicts
        Parameters to vary
    """
    
    
    
    sims = {}
    count = 0
    # for exercise_preset_p in np.arange(0.1, 1.09, 0.1):
    for carb_timeline, pump_carb_timeline in zip(carb_timeline_list, pump_carb_timeline_list):
        t0, patient_config = get_canonical_virtual_patient_model_config(start_glucose_value = start_glucose_value, basal_rate=basal_rate, cir=cir, isf=isf, carb_timeline=carb_timeline, bolus_timeline=bolus_timeline) # patient has many attributes e.g. starting glucose (default: 110), recommendatio accept probability, etc.    
        
        t0, sensor_config = get_canonical_sensor_config(t0, start_value = start_glucose_value) # sensor config has a blood glucose history. right now looks like the starting value repeated 'n' times every 5 minutes before t0
        

        t0, pump_config = get_canonical_risk_pump_config(t0, basal_rate=basal_rate, cir=cir, isf=isf, target_range_min=target_range_min, target_range_max=target_range_max, carb_timeline=pump_carb_timeline, bolus_timeline=bolus_timeline) # sets a carb timeline, bolus timeline (both initialized with 0 at t=0, assuming new values are then added on), basal schedule (e.g. 0.3 units delivered for 24 hours), carb ratio schedule (e.g. constant carb ratio of 20 for 24 hours). similarly for ISR and target range. ques: is there a schedule different from 24 hours?

        patient_config.recommendation_accept_prob = 1.0  # Accept the bolus
        
        sensor_config.std_dev = 1.0 # ques: what does this do

        t0, sim = get_canonical_simulation(
            patient_config=patient_config,
            patient_class=VirtualPatientModel,
            sensor_config=sensor_config,
            # sensor_class=NoisySensor,
            sensor_class=IdealSensor,
            pump_config=pump_config,
            pump_class=ContinuousInsulinPump,
            controller_class=controller,
            multiprocess=True,
            duration_hrs=duration_hrs,
        )

        sim_id = f"sim_{count}"
        sims[sim_id] = sim
        count += 1

    return sims


def plot_insulin_changes(all_results, save_dir):

    br_change = []
    basal = []
    bolus = []
    total_insulin = []

    cgm_mean = []

    fig, ax = plt.subplots(2, 1, sharex=True)
    for sim_id, results_df in all_results.items():
        total_basal_delivered = results_df["delivered_basal_insulin"].sum()
        total_bolus_delivered = results_df["reported_bolus"].sum()

        basal_change = float(re.search("br_change.*(\d+\.\d+).*isf_change", sim_id).groups()[0])
        br_change.append(basal_change)
        basal.append(total_basal_delivered)
        bolus.append(total_bolus_delivered)
        total_insulin.append(total_basal_delivered + total_bolus_delivered)

        cgm_mean.append(results_df["bg_sensor"].mean())

    ax[0].plot(br_change, basal, label="basal")
    ax[0].plot(br_change, bolus, label="bolus")
    ax[0].plot(br_change, total_insulin, label="total")
    plt.legend()
    ax[1].plot(br_change, cgm_mean)
    plt.savefig(save_dir)
    # plt.show()
    


if __name__ == "__main__":
    start = time()
    
    start_glucose_value = 110
    basal_rate = 0.8
    cir = 8.0
    isf = 50.0    
    
    target_range_min = 100
    target_range_max = 120
    
    t0 = DATETIME_DEFAULT
    # carb_timeline = CarbTimeline([t0], [Carb(20, "g", 180)])
    # carb_timeline = CarbTimeline()
    # pump_carb_timeline = CarbTimeline()
    
    # bolus_timeline = BolusTimeline([t0], [Bolus(1.0, "U")])
    bolus_timeline = BolusTimeline()
    
    # controller = DoNothingController
    controller = LoopController
    
    duration_hrs = 8
    
    input_dir = './simulation_inputs'
    save_dir = "./simulation_results/llm_carb_estimate_experiments"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    # load data
    # data_list = ["a_single_serving_final", "lambda_a_multiple_serving_final", "unseen_a_100g_final", "unseen_alternativeUnit_a_single_serving_final", "unseen_alternativeUnit_lambda_a_multiple_serving_final", "unseen_lambda_a_100g_final"]
    data_list = ["a_single_serving_final"]
    model_list = ["gt", "gpt-3.5-turbo", "alpaca-native", "medalpaca-7b"]
    method_list = ["cot_llm"] #, "baseline_llm"]
    
    for data_name in data_list:
        for method_name in method_list:
            fname = f"{data_name}_{method_name}.csv"
            fpath = os.path.join(input_dir, fname)
            
            df = pd.read_csv(fpath)
            print(f"Data: {data_name} | Method: {method_name}")
            print(f"df size: {len(df)}")
            
            df = df[~df.isin([-1]).any(axis=1)] # filter out rows where the prediction for any LLM was -1 (corresponding to "i don't know")
            # df = df.head(100) # select top 10 for debugging
            print(f"df size after filtering: {len(df)}")
            for model_name in model_list:
                mae = abs(df['gt'] - df[model_name]).mean()
                print(f"Running simulations for\nData: {data_name}\nMethod: {method_name}\nModel: {model_name} \nMAE: {mae}")
                
                # true carb timeline comes from the gt
                carb_timeline_list = [CarbTimeline([t0], [Carb(x, "g", 180)]) for x in df['gt']]
                # pump carb timeline comes from the model prediction
                pump_carb_timeline_list = [CarbTimeline([t0], [Carb(x, "g", 180)]) for x in df[model_name]]
                
                sims = build_metabolic_sensitivity_sims(start_glucose_value=start_glucose_value, basal_rate=basal_rate, cir=cir, isf=isf, target_range_min=target_range_min, target_range_max=target_range_max, carb_timeline_list=carb_timeline_list, pump_carb_timeline_list=pump_carb_timeline_list, bolus_timeline=bolus_timeline, duration_hrs=duration_hrs, controller=controller)
                
                all_results, summary_results_df = run_simulations(sims,
                                save_dir=save_dir,
                                save_results=False,
                                num_procs=10)
                
                summary_means = summary_results_df.mean().to_frame().T
                summary_results_df = pd.concat([summary_means, summary_results_df], ignore_index=True)
                
                fname = f"{data_name}_{method_name}_{model_name}.csv"
                summary_results_df.to_csv(os.path.join(save_dir, fname))
    end = time()
    print(f"Time elapsed: {(end-start)/60.0} minutes")
    
    # all_results = load_results(save_dir)

    # plot_sim_results(all_results, n_sims_max_legend=10, save=True, save_path=os.path.join(save_dir, 'sim_results.png'))
    # plot_insulin_changes(all_results, os.path.join(save_dir, 'insulin_changes.png'))
    
    
    # fig1, ax1 = plt.subplots(dpi=150)
    # fig2, ax2 = plt.subplots(dpi=150)

    # for sim_id, results_df in all_results.items():        
    #     preset = float(sim_id[-3:])
    #     bolus_insulin_total = results_df["true_bolus"].sum()
    #     basal_insulin_total = results_df["delivered_basal_insulin"].sum()

    #     entry = {
    #         'preset': preset,
    #         'bolus': bolus_insulin_total,
    #         'basal': basal_insulin_total,
    #         'start_bg': start_glucose_value,
    #         'ISF': isf
    #     }

    #     # results.append(entry)

    #     # get the simulated results
    #     plot_df = results_df.reset_index().iloc[-13 * duration_hrs:]
    #     plot_df.fillna(0, inplace=True)
    #     plot_df["total_delivered"] = plot_df["delivered_basal_insulin"] + plot_df["true_bolus"]
        
    #     ax1.plot(plot_df.time, plot_df.bg, label=sim_id, c=cmap(preset))

    #     ax2.plot(plot_df.time, plot_df.total_delivered, label=sim_id, c=cmap(preset))

    # ax1.legend()
    # ax1.xaxis.set_major_formatter(formatter)
    # ax1.set_xlabel('Time')
    # ax1.set_ylabel('Blood Glucose (mg/dL)')
    # ax1.set_title(f'Starting BG {start_glucose_value}\nISF {isf}')

    # plt.tight_layout()
    # fig1.savefig(f'{save_dir}/bg-bg{start_glucose_value}-isf{isf}.png')

    # plt.close(fig1)

    # ax2.legend()
    # ax2.xaxis.set_major_formatter(formatter)
    # ax2.set_xlabel('Time')
    # ax2.set_ylabel('Total Insulin Delivered (U)')
    # ax2.set_title(f'Starting BG {start_glucose_value}\nISF {isf}')

    # plt.tight_layout()
    # fig2.savefig(f'{save_dir}/insulin-bg{start_glucose_value}-isf{isf}.png')

    # plt.close(fig2)
    
    