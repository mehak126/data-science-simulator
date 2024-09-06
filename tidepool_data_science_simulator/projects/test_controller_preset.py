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
from pandas import Timestamp

from temp_hr import pa_hr

import logging
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.ERROR)

import matplotlib.dates as mdates
formatter = mdates.DateFormatter('%H:%M')
cmap = plt.cm.plasma_r


def build_metabolic_sensitivity_sims(start_glucose_value=110, basal_rate=0.3, cir=20.0, isf=150.0, target_range_min=100, target_range_max=120, carb_timeline=None, pump_carb_timeline=None ,bolus_timeline=None, duration_hrs=1, controller=LoopController, pa_timeline=None, heart_rate_trace=None, exercise_preset_p=1.0):
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
    

    t0, pump_config = get_canonical_risk_pump_config(t0, basal_rate=basal_rate, cir=cir, isf=isf, target_range_min=target_range_min, target_range_max=target_range_max, carb_timeline=pump_carb_timeline, bolus_timeline=bolus_timeline, pa_timeline=pa_timeline) # sets a carb timeline, bolus timeline (both initialized with 0 at t=0, assuming new values are then added on), basal schedule (e.g. 0.3 units delivered for 24 hours), carb ratio schedule (e.g. constant carb ratio of 20 for 24 hours). similarly for ISR and target range. ques: is there a schedule different from 24 hours?

    patient_config.recommendation_accept_prob = 1.0  # Accept the bolus
    
    basal_p_factor = -1 + exercise_preset_p  # note: funky math because of how override function works
    pump_config.basal_schedule.set_override(basal_p_factor)

    isf_p_factor = -1 + 1 / (exercise_preset_p)  # note: funky math because of how override function works
    pump_config.insulin_sensitivity_schedule.set_override(isf_p_factor)

    cir_p_factor = -1 + 1 / (exercise_preset_p)  # note: funky math because of how override function works
    pump_config.carb_ratio_schedule.set_override(cir_p_factor)
    
    sensor_config.std_dev = 1.0

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
    
    save_root = '/home/mdhaliwal/data-science-simulator/tidepool_data_science_simulator/projects/PA_simulations/test_sims/controller_nopreset'

    # controller = DoNothingController
    controller = LoopController
    target_range_min = 100
    target_range_max = 120
    t0 = DATETIME_DEFAULT
    sim_activity_starttime = t0 + timedelta(hours=1) # start activity 1 hour after start of simulation
    
    # no scheduled bolus or carbs
    bolus_timeline = BolusTimeline()
    carb_timeline = CarbTimeline()
    pump_carb_timeline = CarbTimeline()
    
    # preset = 0.6
    preset = 1.0
    hr_val = 124    
    cir = 8.0
    isf = 50
    starting_glucose_vals = [70+i*20 for i in range(12)] 
    egp = 0.8
    pa_duration = 60    
    
    sims = {}
    for starting_glucose in starting_glucose_vals:
        sim_name = f"{cir}_{isf}_{egp}_{starting_glucose}_{pa_duration}"
        savefilename = os.path.join(save_root, f"{sim_name}.tsv")
        if os.path.isfile(savefilename): # check if that sim has already been run
            continue
        
        print(f"CREATING {sim_name}")
        pa_timeline = PhysicalActivityTimeline(datetimes=[sim_activity_starttime], events=[PhysicalActivity(activity='Test Activity', duration=pa_duration)])
        hrs = [hr_val]*(pa_duration*60//10)
        duration_hrs = math.ceil((pa_duration + 2*60) / 60) # total simulation duration is 1 hour before activity + activity duration + 1 hour after activity, rounded up to the neareset exact hour
        sim = build_metabolic_sensitivity_sims(start_glucose_value=starting_glucose, basal_rate=egp, cir=cir, isf=isf, target_range_min=target_range_min, target_range_max=target_range_max, carb_timeline=carb_timeline, pump_carb_timeline=pump_carb_timeline, bolus_timeline=bolus_timeline, duration_hrs=duration_hrs, controller=controller, pa_timeline=pa_timeline, heart_rate_trace = hrs, exercise_preset_p=preset)
        sims[sim_name] = sim  
            
    print(f"RUNNING {len(sims)} sims")
    save_dir = save_root
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    batch_size = 100 # run sims in batches to avoid "too many open files" error
    sims_items = list(sims.items())
    sim_batches = [dict(sims_items[i:i + batch_size]) for i in range(0, len(sims), batch_size)]

    for batch_num, sim_batch in enumerate(sim_batches):
        print(f"BATCH {batch_num} of {len(sim_batches)}")
        all_results, summary_results_df = run_simulations(sim_batch,
                    save_dir=save_dir,
                    save_results=True,
                    num_procs=10, name = f'{batch_num}')    
    
        for sim_id, results_df in all_results.items():
            hr_trace = sims[sim_id].virtual_patient.patient_config.hr_trace
            hr_series = pd.Series(data=hr_trace.hr_values, index=hr_trace.datetimes)
            filtered_hrs_series = hr_series.loc[hr_series.index.isin(results_df.index)]
            results_df['hrs'] = filtered_hrs_series.reindex(results_df.index)                                
            results_df_dict = {sim_id: results_df}
            plot_sim_results(results_df_dict, n_sims_max_legend=10, save=True, save_path=os.path.join(save_dir, f'{save_dir}/{sim_id}.png'))
    print(f"COMPLETED ALL SIMS")    