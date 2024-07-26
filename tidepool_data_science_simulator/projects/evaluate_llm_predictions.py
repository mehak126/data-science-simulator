from ftplib import FTP_PORT
import sys
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../..')

import copy
import os
import re
import json
import random
random.seed(42)

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


def build_metabolic_sensitivity_sims(start_glucose_value=110, basal_rate=0.3, cir=20.0, isf=150.0, target_range_min=100, target_range_max=120, carb_timeline_list=[], pump_carb_timeline_list=[] ,bolus_timeline_list=[], duration_hrs=1, controller=LoopController):
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
    for carb_timeline, pump_carb_timeline, bolus_timeline in zip(carb_timeline_list, pump_carb_timeline_list, bolus_timeline_list):
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
    
    
    controller = DoNothingController
    # controller = LoopController
    
    duration_hrs = 8
    
    save_dir = "./simulation_results/human_llm_eval"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    # data_list = [
    #     "a_single_serving_final",
    #     "lambda_a_multiple_serving_final",
    #     "unseen_a_100g_final",
    #     "unseen_alternativeUnit_a_single_serving_final",
    #     "unseen_alternativeUnit_lambda_a_multiple_serving_final",
    #     "unseen_lambda_a_100g_final",
    #     "a_b_single_serving_final",
    #     "lambda_a_beta_b_multiple_serving_final",
    #     "unseen_a_b_100g_final",
    #     "unseen_lambda_a_beta_b_100g_final",
    #     "unseen_alternativeUnit_a_b_single_serving_final",
    #     "unseen_alternativeUnit_lambda_a_b_multiple_serving_final",
    #     "a_b_c_single_serving_final",
    #     "unseen_alternativeUnit_a_b_c_single_serving_final",
    #     "unseen_a_b_c_100g_final"
    # ]
    
    model_list = ["gpt-3.5-turbo"] #, "alpaca-native", "medalpaca-7b"] #, "Llama-2-7b-chat-hf", "Llama-2-70b-chat-hf", "Meta-Llama-3-70B-Instruct", "Meta-Llama-3-8B-Instruct"]
    # method_list = ["baseline_llm", "cot_llm", "generate_parsed_query_baseline", "generate_parsed_query_cot"]
    method_list = ["cot_llm"]
    simulation_inp_dir = "/home/mdhaliwal/data-science-simulator/tidepool_data_science_simulator/projects/human_simulation_inputs"
    
    for method_in, method_name in enumerate(method_list):
        # print(f"Method: {method_name}")
        for model_in, model_name in enumerate(model_list):
            # print(f"Model: {model_name}")        
            # for data_in, data_name in enumerate(data_list):
                # fname = f"{model_name}_{method_name}_{data_name}.csv"
                fname = f"{model_name}_{method_name}.csv"
                fpath = os.path.join(simulation_inp_dir, fname)
                if not os.path.exists(fpath):
                    print(f"{fname} NA")
                    continue
                inp_df = pd.read_csv(fpath)
                inp_df = inp_df[~inp_df.apply(lambda row: row.isin([-1]).any(), axis=1)]
                # inp_df = inp_df.head(10)
                if len(inp_df) == 0:
                    print(f"{fname} SIZE = 0")
                    continue

                # true carb timeline
                true_carb_vals = list(inp_df['gt'])
                carb_timeline_list = [CarbTimeline([t0], [Carb(x, "g", 180)]) for x in true_carb_vals]
                
                # predicted carb timeline
                pred_carb_vals = list(inp_df['pred'])
                pump_carb_timeline_list = [CarbTimeline([t0], [Carb(x, "g", 180)]) for x in pred_carb_vals]                
                
                # bolus timeline
                bolus_timeline_list = [BolusTimeline([t0], [Bolus(x/cir, "U")]) for x in pred_carb_vals]
                                
                sims = build_metabolic_sensitivity_sims(start_glucose_value=start_glucose_value, basal_rate=basal_rate, cir=cir, isf=isf, target_range_min=target_range_min, target_range_max=target_range_max, carb_timeline_list=carb_timeline_list, pump_carb_timeline_list=pump_carb_timeline_list, bolus_timeline_list=bolus_timeline_list, duration_hrs=duration_hrs, controller=controller)
                
                all_results, summary_results_df = run_simulations(sims,
                    save_dir=save_dir,
                    save_results=False,
                    num_procs=10)
                
                summary_results_df['gt'] = true_carb_vals
                summary_results_df['pred'] = pred_carb_vals
                summary_results_df['scheduled_bolus'] = [x/cir for x in pred_carb_vals]
                fname = f"{model_name}_{method_name}_{data_name}.csv"
                print(f"Saving: {fname}")
                summary_results_df.to_csv(os.path.join(save_dir, fname))                
                
                if method_in == 0 and model_in == 0:
                    # true carb timeline
                    true_carb_vals = list(inp_df['gt'])
                    carb_timeline_list = [CarbTimeline([t0], [Carb(x, "g", 180)]) for x in true_carb_vals]
                    # predicted carb timeline
                    pred_carb_vals = list(inp_df['gt'])
                    pump_carb_timeline_list = [CarbTimeline([t0], [Carb(x, "g", 180)]) for x in pred_carb_vals]                                    
                    # bolus timeline
                    bolus_timeline_list = [BolusTimeline([t0], [Bolus(x/cir, "U")]) for x in pred_carb_vals]
                    sims = build_metabolic_sensitivity_sims(start_glucose_value=start_glucose_value, basal_rate=basal_rate, cir=cir, isf=isf, target_range_min=target_range_min, target_range_max=target_range_max, carb_timeline_list=carb_timeline_list, pump_carb_timeline_list=pump_carb_timeline_list, bolus_timeline_list=bolus_timeline_list, duration_hrs=duration_hrs, controller=controller)
                
                    all_results, summary_results_df = run_simulations(sims,
                        save_dir=save_dir,
                        save_results=False,
                        num_procs=10)
                    
                    summary_results_df['gt'] = true_carb_vals
                    summary_results_df['pred'] = pred_carb_vals
                    summary_results_df['scheduled_bolus'] = [x/cir for x in pred_carb_vals]
                    fname = f"gt_{data_name}.csv"
                    print(f"Saving: {fname}")
                    summary_results_df.to_csv(os.path.join(save_dir, fname)) 
                    
                    #    random guess
                    # true carb timeline
                    true_carb_vals = list(inp_df['gt'])
                    carb_timeline_list = [CarbTimeline([t0], [Carb(x, "g", 180)]) for x in true_carb_vals]
                    # predicted carb timeline
                    pred_carb_vals = [random.uniform(30, 75) for _ in true_carb_vals]
                    pump_carb_timeline_list = [CarbTimeline([t0], [Carb(x, "g", 180)]) for x in pred_carb_vals]                                    
                    # bolus timeline
                    bolus_timeline_list = [BolusTimeline([t0], [Bolus(x/cir, "U")]) for x in pred_carb_vals]
                    sims = build_metabolic_sensitivity_sims(start_glucose_value=start_glucose_value, basal_rate=basal_rate, cir=cir, isf=isf, target_range_min=target_range_min, target_range_max=target_range_max, carb_timeline_list=carb_timeline_list, pump_carb_timeline_list=pump_carb_timeline_list, bolus_timeline_list=bolus_timeline_list, duration_hrs=duration_hrs, controller=controller)
                
                    all_results, summary_results_df = run_simulations(sims,
                        save_dir=save_dir,
                        save_results=False,
                        num_procs=10)
                    
                    summary_results_df['gt'] = true_carb_vals
                    summary_results_df['pred'] = pred_carb_vals
                    summary_results_df['scheduled_bolus'] = [x/cir for x in pred_carb_vals]
                    fname = f"random_{data_name}.csv"
                    print(f"Saving: {fname}")
                    summary_results_df.to_csv(os.path.join(save_dir, fname)) 
                            
  
    end = time()
    print(f"Time elapsed: {(end-start)/60.0} minutes")
    
    