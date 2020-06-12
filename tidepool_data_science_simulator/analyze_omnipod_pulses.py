__author__ = "Cameron Summers"

__author__ = "Cameron Summers"

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tidepool_data_science_simulator.makedata.make_patient import get_canonical_risk_patient
from tidepool_data_science_simulator.makedata.make_simulation import get_canonical_simulation
from tidepool_data_science_simulator.models.measures import TempBasal, BasalRate

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

from tidepool_data_science_simulator.models.simulation import (
    Simulation, SettingSchedule24Hr, BasalSchedule24hr, TargetRangeSchedule24hr
)
from tidepool_data_science_simulator.models.controller import DoNothingController, LoopController
from tidepool_data_science_simulator.models.patient import VirtualPatient
from tidepool_data_science_simulator.models.pump import OmnipodMissingPulses, Omnipod, ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import IdealSensor, NoisySensor
from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results_missing_insulin
from tidepool_data_science_simulator.utils import timing

from tidepool_data_science_metrics.insulin.insulin import dka_index, dka_risk_score


def analyze_omnipod_missing_pulses():
    current_time, patient = get_canonical_risk_patient(pump_class=OmnipodMissingPulses)
    pump = patient.pump

    delivered_insulin = []
    undelivered_insulin = []

    temp_basal_scenario = [0.3, 0.1] * 6  # This scenario gives no insulin

    update_delta = datetime.timedelta(minutes=5)
    for temp_basal_value in temp_basal_scenario:
        temp_basal = TempBasal(pump.time, temp_basal_value, duration_minutes=30, units="U/hr")
        pump.set_temp_basal(temp_basal)

        pump.update(pump.time + update_delta)

        delivered_insulin.append(patient.pump.basal_insulin_delivered_last_update)
        undelivered_insulin.append(patient.pump.basal_undelivered_insulin_since_last_update)

    total_delivered_insulin = np.sum(delivered_insulin)
    total_undelivered_insulin = np.sum(undelivered_insulin)
    total_expected_insulin = total_delivered_insulin + total_undelivered_insulin
    print("Total Delivered Insulin  {:>4} ({:.0f}%)".format(total_delivered_insulin,
                                                            total_delivered_insulin / total_expected_insulin * 100.0)),
    print("Total Undelivered Insulin {:>4} ({:.0f}%)".format(total_undelivered_insulin,
                                                             total_undelivered_insulin / total_expected_insulin * 100.0))

    plt.title("Omnipod Missing Insulin Pulses")
    plt.xlabel("Time Step (5 min)")
    plt.ylabel("Insulin (U or U/hr)")
    plt.plot(delivered_insulin, label="delivered", marker=".")
    plt.plot(undelivered_insulin, label="undelivered", marker='.')
    plt.plot(temp_basal_scenario, label="Temp Basal Values", marker=".")
    plt.legend()
    plt.show()


def analyze_omnipod_missing_insulin_across_basal_rates():
    basal_rates = np.arange(0.0, 3.0, 0.05)

    all_delivered_insulin = []
    all_undelivered_insulin = []

    for br in basal_rates:
        current_time, patient = get_canonical_risk_patient(pump_class=OmnipodMissingPulses)
        pump = patient.pump

        delivered_basal_insulin = 0
        undelivered_basal_insulin = 0

        update_delta = datetime.timedelta(minutes=5)
        for _ in range(12):
            temp_basal = TempBasal(pump.time, br, 30, "U/hr")
            pump.set_temp_basal(temp_basal)
            pump.update(pump.time + update_delta)

            delivered_basal_insulin += patient.pump.basal_insulin_delivered_last_update
            undelivered_basal_insulin += patient.pump.basal_undelivered_insulin_since_last_update

        all_delivered_insulin.append(delivered_basal_insulin)
        all_undelivered_insulin.append(undelivered_basal_insulin)

    plt.plot(basal_rates, all_delivered_insulin, label="Delivered Basal", marker='.')
    plt.plot(basal_rates, all_undelivered_insulin, label="Undelivered Basal", marker='.')
    plt.xlabel("Temp Basal Rate (U/hr)")
    plt.ylabel("Total Basal Insulin over 1 Hr (U)")
    plt.title("Constant Temp Basal Missing Insulin")
    plt.legend()
    plt.show()


@timing
def analyze_omnipod_missing_pulses_wLoop(dry_run):
    """
    Compare two controllers for a given scenario file:
        1. No controller, ie no insulin modulation except for pump schedule
        2. Loop controller

    Parameters
    ----------
    scenario_csv_filepath: str
        Path to the scenario file
    """

    param_grid = [
        {
            "loop_max_basal_rate": round(sbr * xer, 2),
            "patient_basal_rate": round(sbr, 2),
            "pump_basal_rate": round(sbr, 2)
        }
        # for sbr in np.arange(0.05, 0.75, 0.05)
        # for xer in [1.5, 2, 3, 4, 5, 6] #range(2, 20)
        for sbr in [0.2, 0.3]
        for xer in [2, 3]
    ]
    sim_num_hours = 24

    if dry_run:
        sim_num_hours = 2
        param_grid = param_grid[:1]

    sims = {}
    sim_params = {}
    for pgrid in param_grid:

        np.random.seed(1234)

        t0, sim = get_canonical_simulation(
            sensor_class=NoisySensor,
            pump_class=OmnipodMissingPulses,
            controller_class=LoopController,
            multiprocess=True,
            duration_hrs=sim_num_hours
        )

        sim_id = "SBR {pump_basal_rate} VPBR {patient_basal_rate} MBR {loop_max_basal_rate}".format(**pgrid)
        print("Running: {}".format(sim_id))

        patient_config = sim.virtual_patient.patient_config
        patient_config.recommendation_accept_prob = 0.0  # TODO: put in scenario file

        patient_config.basal_schedule = BasalSchedule24hr(t0,
                                                          [datetime.time(hour=0, minute=0, second=0)],
                                                          [BasalRate(pgrid['patient_basal_rate'], "U/hr")],
                                                          [1440])

        pump_config = sim.virtual_patient.pump.pump_config
        pump_config.basal_schedule = BasalSchedule24hr(t0,
                                                       [datetime.time(hour=0, minute=0, second=0)],
                                                       [BasalRate(pgrid['pump_basal_rate'], "U/hr")],
                                                       [1440])

        controller_config = sim.controller.controller_config
        controller_config.controller_settings["max_basal_rate"] = pgrid["loop_max_basal_rate"]

        sims[sim_id] = sim
        sim_params[sim_id] = pgrid
        sim.start()

    all_results = {id: sim.queue.get() for id, sim in sims.items()}
    [sim.join() for id, sim in sims.items()]

    summary_results_df = []
    for sim_id, results_df in all_results.items():
        dkai = dka_index(results_df['iob'], sim_params[sim_id]["patient_basal_rate"])
        dkars = dka_risk_score(dkai)

        print("dkai", dkai)
        row = {
            "dka_index": dkai,
            "dka_risk_score": dkars,
            "loop_max_basal_rate": sim_params[sim_id]["loop_max_basal_rate"],
            "sbr": sim_params[sim_id]["patient_basal_rate"]
        }
        summary_results_df.append(row)

    summary_results_df = pd.DataFrame(summary_results_df)

    if 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        summary_results_pivot_df = summary_results_df.pivot(index='sbr', columns='loop_max_basal_rate',
                                                            values='dka_index')
        sns.heatmap(summary_results_pivot_df, ax=ax1)
        ax1.set_title("DKAI for Canonical Patient with Low Basal Settings")

        plt.figure()
        summary_results_pivot_df = summary_results_df.pivot(index='sbr', columns='loop_max_basal_rate',
                                                            values='dka_risk_score')
        sns.heatmap(summary_results_pivot_df, ax=ax2)
        ax2.set_title("DKA Risk Score for Canonical Patient with Low Basal Settings")

        plt.show()
    else:
        plot_sim_results_missing_insulin(all_results)


if __name__ == "__main__":

    # analyze_omnipod_missing_pulses_wLoop(False)
    analyze_omnipod_missing_pulses()
    # analyze_omnipod_missing_insulin_across_basal_rates()
