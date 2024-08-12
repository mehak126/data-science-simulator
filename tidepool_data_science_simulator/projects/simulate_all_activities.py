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
import math

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
from datetime import datetime, timedelta

from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_simulator.run import run_simulations

from tidepool_data_science_simulator.models.events import CarbTimeline, BolusTimeline, PhysicalActivityTimeline
from tidepool_data_science_simulator.models.measures import Carb, Bolus, PhysicalActivity

import datetime

from temp_hr import pa_hr

import logging
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.ERROR)

import matplotlib.dates as mdates
formatter = mdates.DateFormatter('%H:%M')
cmap = plt.cm.plasma_r


def build_metabolic_sensitivity_sims(start_glucose_value=110, basal_rate=0.3, cir=20.0, isf=150.0, target_range_min=100, target_range_max=120, carb_timeline=None, pump_carb_timeline=None ,bolus_timeline=None, duration_hrs=1, controller=LoopController, pa_timeline=None, heart_rate_trace=None):
    """
    Look at resulting bgs from settings that are correct/incorrect for analysis.

    Parameters
    ----------
    scenario_csv_filepath: str
        Path to the scenario file

    param_grid: list of dicts
        Parameters to vary
    """
    
    t0, patient_config = get_canonical_virtual_patient_model_config(start_glucose_value = start_glucose_value, basal_rate=basal_rate, cir=cir, isf=isf, carb_timeline=carb_timeline, bolus_timeline=bolus_timeline, pa_timeline=pa_timeline, sim_length=duration_hrs, heart_rate_trace=heart_rate_trace) # patient has many attributes e.g. starting glucose (default: 110), recommendatio accept probability, etc.    
    
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
    
    return sim
    


if __name__ == "__main__":
    
    # these parameters remain constant in all simulations
    basal_rate = 0.8
    cir = 8.0
    isf = 50.0    
    
    target_range_min = 100
    target_range_max = 120
    
    t0 = DATETIME_DEFAULT
    
    carb_timeline = CarbTimeline()
    pump_carb_timeline = CarbTimeline()
    bolus_timeline = BolusTimeline()
    
    controller = DoNothingController
    # controller = LoopController
    
    activity_names = activity_names = ['Walking, Dog Walking', 'Biking (Indoor or Outdoor)', 'Jogging/Running (Indoor or Outdoor)', 'Strength Training, Weight Lifting']
    
    root = '/home/mdhaliwal/data-science-simulator/tidepool_data_science_simulator/projects/PA_simulations/T1DEXI_data/'
    # final_dir = 'PA_only'
    final_dir = 'PA_1hr_after'
    
    save_root = '/home/mdhaliwal/data-science-simulator/tidepool_data_science_simulator/projects/PA_simulations/simulation_data/'
    
    for activity in activity_names:
        print(f"RUNNING SIMULATIONS FOR {activity}")
        activity = activity.split('/')[0]
        fname = f"data_{activity}.csv"
        fpath = os.path.join(root, final_dir, fname)
        df = pd.read_csv(fpath)
        # df = df.head(200)
        all_cgms = df['cgm']
        all_hrs = df['hr']
        all_durations = df['duration']
        pa_type = activity
        
        
        index = 0
        sims = {}
        for duration, cgms, hrs in zip(all_durations, all_cgms, all_hrs):
            cgms = eval(cgms)
            hrs = eval(hrs)
            
            # these parameters vary for each session
            start_glucose_value = cgms[0]
            pa_duration = int(duration) # each cgm value represents 5 minutes
            pa_timeline = PhysicalActivityTimeline(datetimes=[t0], events=[PhysicalActivity(activity=pa_type, duration=pa_duration)])
            duration_hrs = math.ceil(duration / 60)
            
            sim = build_metabolic_sensitivity_sims(start_glucose_value=start_glucose_value, basal_rate=basal_rate, cir=cir, isf=isf, target_range_min=target_range_min, target_range_max=target_range_max, carb_timeline=carb_timeline, pump_carb_timeline=pump_carb_timeline, bolus_timeline=bolus_timeline, duration_hrs=duration_hrs, controller=controller, pa_timeline=pa_timeline, heart_rate_trace = hrs)
            sims[f"{activity}_{index}"] = sim
            index += 1
            
            
        print(f"RUNNING {len(sims)} sims")
        save_dir = os.path.join(save_root, final_dir)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        batch_size = 100
        sims_items = list(sims.items())
        sim_batches = [dict(sims_items[i:i + batch_size]) for i in range(0, len(sims), batch_size)]

        for batch_num, sim_batch in enumerate(sim_batches):
            print(f"BATCH {batch_num} of {len(sim_batches)}")
            all_results, summary_results_df = run_simulations(sim_batch,
                        save_dir=save_dir,
                        save_results=True,
                        num_procs=10)        

        
            for sim_id, results_df in all_results.items():        
                # get the simulated results
                plot_df = results_df.reset_index().iloc[-13 * duration_hrs:]
                plot_df.fillna(0, inplace=True)
                plot_df["total_delivered"] = plot_df["delivered_basal_insulin"] + plot_df["true_bolus"]
                
                # new plot mehak
                # Create the figure and the left y-axis
                fig, ax1 = plt.subplots()

                # Plot CGM values
                ax1.plot(plot_df.time, plot_df.bg, 'g-', label='CGM Values (mg/dl)')
                ax1.set_xlabel('Timestamp')
                ax1.set_ylabel('CGM Values', color='g')
                ax1.tick_params(axis='y', labelcolor='g')
                ax1.tick_params(axis='x', rotation=90)

                # Create the right y-axis
                ax2 = ax1.twinx()

                # Plot HR values
                hr_trace = sims[sim_id].virtual_patient.patient_config.hr_trace
                hr_datetimes = hr_trace.datetimes
                hr_vals = hr_trace.hr_values
                
                ax2.plot(hr_datetimes, hr_vals, 'b-', label='HR Values (BPM)')
                ax2.set_ylabel('HR Values', color='b')
                ax2.tick_params(axis='y', labelcolor='b')

                # Add a legend to the plot
                fig.legend(loc="upper left", bbox_to_anchor=(0,1), bbox_transform=ax1.transAxes)

                # save the plot
                plt.tight_layout()
                # fig.savefig(f'{save_dir}/bg-bg{start_glucose_value}-isf{isf}.png')
                fig.savefig(f'{save_dir}/{sim_id}.png')
                plt.close(fig)
            
            
