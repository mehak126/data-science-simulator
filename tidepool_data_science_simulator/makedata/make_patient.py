__author__ = "Cameron Summers"

import datetime
import numpy as np

import logging

logger = logging.getLogger(__name__)

from tidepool_data_science_simulator.models.simulation import (
    SettingSchedule24Hr, BasalSchedule24hr, TargetRangeSchedule24hr
 
)
from tidepool_data_science_simulator.models.events import CarbTimeline, BolusTimeline, ActionTimeline
from tidepool_data_science_simulator.makedata.scenario_parser import PumpConfig, PatientConfig, SensorConfig
from tidepool_data_science_simulator.models.measures import (
    InsulinSensitivityFactor, GlucoseSensitivityFactor, BasalBloodGlucose, InsulinProductionRate,
    CarbInsulinRatio, BasalRate, TargetRange, GlucoseTrace, Bolus, Carb
)
from tidepool_data_science_simulator.models.pump import OmnipodMissingPulses, Omnipod, ContinuousInsulinPump
from tidepool_data_science_simulator.models.patient import VirtualPatient, VirtualPatientModel
from tidepool_data_science_simulator.models.sensor import IdealSensor

from tidepool_data_science_simulator.models.controller import LoopController

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

SINGLE_SETTING_START_TIME = datetime.time(hour=0, minute=0, second=0)
SINGLE_SETTING_DURATION = 1440
DATETIME_DEFAULT = datetime.datetime(year=2019, month=8, day=15, hour=12, minute=0, second=0)


def get_canonical_glucose_history(t0, num_glucose_values=137, start_value=110):
    """
    Common glucose history.

    Parameters
    ----------
    t0 (datetime.datetime): time of last glucose value
    num_glucose_values (int): num glucose values

    Returns
    -------
        GlucoseTrace
    """

    true_bg_values_history = [start_value] * num_glucose_values

    true_bg_dates = [t0 - datetime.timedelta(minutes=i * 5) for i in range(num_glucose_values)]
    true_bg_dates.reverse()
    true_bg_history = GlucoseTrace(true_bg_dates, true_bg_values_history)

    return true_bg_history


def get_canonical_risk_pump_config(t0=DATETIME_DEFAULT):
    """
    Get canonical pump config

    Parameters
    ----------
    t0

    Returns
    -------
    PumpConfig
    """

    pump_carb_timeline = CarbTimeline([t0], [Carb(0.0, "g", 180)])
    pump_bolus_timeline = BolusTimeline([t0], [Bolus(0.0, "U")])

    pump_config = PumpConfig(
        basal_schedule=BasalSchedule24hr(
            t0,
            start_times=[SINGLE_SETTING_START_TIME],
            values=[BasalRate(0.3, "U/hr")],
            duration_minutes=[SINGLE_SETTING_DURATION]
        ),
        carb_ratio_schedule=SettingSchedule24Hr(
            t0,
            "CIR",
            start_times=[SINGLE_SETTING_START_TIME],
            values=[CarbInsulinRatio(20.0, "g/U")],
            duration_minutes=[SINGLE_SETTING_DURATION]
        ),
        insulin_sensitivity_schedule=SettingSchedule24Hr(
            t0,
            "ISF",
            start_times=[SINGLE_SETTING_START_TIME],
            values=[InsulinSensitivityFactor(150.0, "mg/dL/U")],
            duration_minutes=[SINGLE_SETTING_DURATION]
        ),
        target_range_schedule=TargetRangeSchedule24hr(
            t0,
            start_times=[SINGLE_SETTING_START_TIME],
            values=[TargetRange(100, 120, "mg/dL")],
            duration_minutes=[SINGLE_SETTING_DURATION]
        ),
        carb_event_timeline=pump_carb_timeline,
        bolus_event_timeline=pump_bolus_timeline
    )

    return t0, pump_config

# def get_canonical_timeline_risk_pump_config(t0=DATETIME_DEFAULT):
#     """
#     Get canonical pump config using timelines instead of 24 hour schedules
#     This is necessary for using the Swift version of Loop

#     Parameters
#     ----------
#     t0

#     Returns
#     -------
#     PumpConfig
#     """
    
#     pump_carb_timeline = CarbTimeline([t0], [Carb(0.0, "g", 180)])
#     pump_bolus_timeline = BolusTimeline([t0], [Bolus(0.0, "U")])
#     dt = datetime.datetime(year=2019, month=8, day=15, hour=0, minute=0, second=0)
    
#     pump_config = PumpConfig(
#         basal_schedule=BasalTimeline(
#             t0,
#             start_times=[dt],
#             values=[BasalRate(0.3, "U/hr")],
#             duration_minutes=[SINGLE_SETTING_DURATION * 2]
#         ),
#         carb_ratio_schedule=SettingTimeline(
#             t0,
#             "CIR",
#             start_times=[dt],
#             values=[CarbInsulinRatio(20.0, "g/U")],
#             duration_minutes=[SINGLE_SETTING_DURATION * 2]
#         ),
#         insulin_sensitivity_schedule=SettingTimeline(
#             t0,
#             "ISF",
#             start_times=[dt],
#             values=[InsulinSensitivityFactor(150.0, "mg/dL/U")],
#             duration_minutes=[SINGLE_SETTING_DURATION * 2]
#         ),
#         target_range_schedule=TargetRangeTimeline(
#             t0,
#             start_times=[dt],
#             values=[TargetRange(100, 120, "mg/dL")],
#             duration_minutes=[SINGLE_SETTING_DURATION * 2]
#         ),
#         carb_event_timeline=pump_carb_timeline,
#         bolus_event_timeline=pump_bolus_timeline
#     )

#     return t0, pump_config

def get_canonical_sensor_config(t0=DATETIME_DEFAULT, num_glucose_values=137, start_value=110):
    """
    Get canonical sensor config

    Parameters
    ----------
    t0

    Returns
    -------
    SensorConfig
    """
    sensor_bg_history = get_canonical_glucose_history(t0, num_glucose_values=num_glucose_values, start_value=start_value)
    sensor_config = SensorConfig(sensor_bg_history)
    return t0, sensor_config


def get_canonical_risk_patient_config(t0=DATETIME_DEFAULT, start_glucose_value=110):
    """
    Get canonical patient config

    Parameters
    ----------
    t0

    Returns
    -------
    PatientConfig
    """

    patient_carb_timeline = CarbTimeline([t0], [Carb(0.0, "g", 180)])
    patient_bolus_timeline = BolusTimeline([t0], [Bolus(0.0, "U")])

    true_bg_history = get_canonical_glucose_history(t0, start_value=start_glucose_value)

    patient_config = PatientConfig(
        basal_schedule=BasalSchedule24hr(
            t0,
            start_times=[SINGLE_SETTING_START_TIME],
            values=[BasalRate(0.3, "mg/dL")],
            duration_minutes=[SINGLE_SETTING_DURATION]
        ),
        carb_ratio_schedule=SettingSchedule24Hr(
            t0,
            "CIR",
            start_times=[SINGLE_SETTING_START_TIME],
            values=[CarbInsulinRatio(20.0, "g/U")],
            duration_minutes=[SINGLE_SETTING_DURATION]
        ),
        insulin_sensitivity_schedule=SettingSchedule24Hr(
            t0,
            "ISF",
            start_times=[SINGLE_SETTING_START_TIME],
            values=[InsulinSensitivityFactor(150.0, "md/dL / U")],
            duration_minutes=[SINGLE_SETTING_DURATION]
        ),
        glucose_history=true_bg_history,
        carb_event_timeline=patient_carb_timeline,
        bolus_event_timeline=patient_bolus_timeline,
        action_timeline=ActionTimeline(),
    )

    patient_config.recommendation_accept_prob = 0.0  # Does not accept any bolus recommendations
    patient_config.min_bolus_rec_threshold = 0.5  # Minimum size of bolus to accept
    patient_config.recommendation_meal_attention_time_minutes = 1e12  # Time since meal to take recommendations

    return t0, patient_config


def get_canonical_risk_patient(t0=DATETIME_DEFAULT,
                               patient_class=VirtualPatient,
                               patient_config=None,
                               pump_class=ContinuousInsulinPump,
                               pump_config=None,
                               sensor_class=IdealSensor,
                               sensor_config=None,
                               ):
    """
    Get patient with settings from the FDA risk analysis scenario template.
    Putting here to simplify testing and decouple from specific scenarios.

    Returns
    -------
    (datetime.datetime, VirtualPatient)
        Current time of patient, Canonical patient
    """

    if pump_config is None:
        t0, pump_config = get_canonical_risk_pump_config(t0)

    if patient_config is None:
        t0, patient_config = get_canonical_risk_patient_config(t0)

    if sensor_config is None:
        t0, sensor_config = get_canonical_sensor_config(t0)

    pump = pump_class(
        time=t0,
        pump_config=pump_config
    )

    sensor = sensor_class(
        time=t0,
        sensor_config=sensor_config
    )

    virtual_patient = patient_class(
        time=t0,
        pump=pump,
        sensor=sensor,
        metabolism_model=SimpleMetabolismModel,
        patient_config=patient_config,
    )

    return t0, virtual_patient


def get_variable_risk_patient_config(random_state, t0=DATETIME_DEFAULT):
    """
    Get canonical patient config

    Parameters
    ----------
    t0

    Returns
    -------
    PatientConfig
    """
    patient_carb_timeline = CarbTimeline([t0], [Carb(0.0, "g", 180)])
    patient_bolus_timeline = BolusTimeline([t0], [Bolus(0.0, "U")])

    true_bg_history = get_canonical_glucose_history(t0)

    total_hours_in_day = 24
    hours_in_day = range(total_hours_in_day)
    carb_gain = random_state.uniform(7, 9)
    basal_rates = [random_state.uniform(0.2, 0.4) for _ in hours_in_day]
    carb_ratios = [random_state.uniform(18, 22) for _ in hours_in_day]
    isfs = [carb_gain * carb_ratio for carb_ratio in carb_ratios]

    start_times_hourly = [datetime.time(hour=i, minute=0, second=0) for i in hours_in_day]
    durations_min_per_hour = [SINGLE_SETTING_DURATION / total_hours_in_day for _ in hours_in_day]

    patient_config = PatientConfig(
        basal_schedule=BasalSchedule24hr(
            t0,
            start_times=start_times_hourly,
            values=[BasalRate(rate, "mg/dL") for rate in basal_rates],
            duration_minutes=durations_min_per_hour
        ),
        carb_ratio_schedule=SettingSchedule24Hr(
            t0,
            "CIR",
            start_times=start_times_hourly,
            values=[CarbInsulinRatio(carb_ratio, "g/U") for carb_ratio in carb_ratios],
            duration_minutes=durations_min_per_hour
        ),
        insulin_sensitivity_schedule=SettingSchedule24Hr(
            t0,
            "ISF",
            start_times=start_times_hourly,
            values=[InsulinSensitivityFactor(isf, "md/dL / U") for isf in isfs],
            duration_minutes=durations_min_per_hour
        ),
        glucose_history=true_bg_history,
        carb_event_timeline=patient_carb_timeline,
        bolus_event_timeline=patient_bolus_timeline,
        action_timeline=ActionTimeline(),
    )

    patient_config.recommendation_accept_prob = 0.0  # Does not accept any bolus recommendations
    patient_config.min_bolus_rec_threshold = 0.5  # Minimum size of bolus to accept
    patient_config.recommendation_meal_attention_time_minutes = 1e12  # Time since meal to take recommendations

    return t0, patient_config


def get_canonical_virtual_patient_model_config(random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(0)

    t0, patient_config = get_canonical_risk_patient_config()

    patient_config.recommendation_accept_prob = random_state.uniform(0.8, 0.99)
    patient_config.min_bolus_rec_threshold = random_state.uniform(0.4, 0.6)
    patient_config.correct_bolus_bg_threshold = random_state.uniform(140, 190)  # no impact
    patient_config.correct_bolus_delay_minutes = random_state.uniform(20, 40)  # no impact
    patient_config.correct_carb_bg_threshold = random_state.uniform(70, 90)
    patient_config.correct_carb_delay_minutes = random_state.uniform(5, 15)
    patient_config.carb_count_noise_percentage = random_state.uniform(0.1, 0.25)
    patient_config.report_bolus_probability = random_state.uniform(1.0, 1.0)  # no impact
    patient_config.report_carb_probability = random_state.uniform(0.95, 1.0)
    patient_config.recommendation_meal_attention_time_minutes = np.inf

    patient_config.prebolus_minutes_choices = [0]
    patient_config.carb_reported_minutes_choices = [0]

    return t0, patient_config


def get_pump_config_from_patient(random_state, patient_config, risk_level=0, t0=DATETIME_DEFAULT):
    """
    Get pump config that is the mean of the true patient config values.

    Parameters
    ----------
    t0

    Returns
    -------
    PumpConfig
    """
    pump_carb_timeline = CarbTimeline([t0], [Carb(0.0, "g", 180)])
    pump_bolus_timeline = BolusTimeline([t0], [Bolus(0.0, "U")])

    # Risk Configurable values
    patient_basal_values = [br.value for br in patient_config.basal_schedule.schedule.values()]
    patient_basal_mean = np.mean(patient_basal_values)
    patient_basal_std = np.std(patient_basal_values)

    patient_cir_values = [cir.value for cir in patient_config.carb_ratio_schedule.schedule.values()]
    patient_cir_mean = np.mean(patient_cir_values)
    patient_cir_std = np.std(patient_cir_values)

    patient_isf_values = [isf.value for isf in patient_config.insulin_sensitivity_schedule.schedule.values()]
    patient_isf_mean = np.mean(patient_isf_values)
    patient_isf_std = np.std(patient_isf_values)

    pump_br = random_state.normal(patient_basal_mean, patient_basal_std * risk_level)
    pump_cir = random_state.normal(patient_cir_mean, patient_cir_std * risk_level)
    pump_isf = random_state.normal(patient_isf_mean, patient_isf_std * risk_level)

    logger.debug("Basal Patient {:.2f} Pump {:.2f}".format(patient_basal_mean, pump_br))
    logger.debug("CIR Patient {:.2f} Pump {:.2f}".format(patient_cir_mean, pump_cir))
    logger.debug("ISF Patient {:.2f} Pump {:.2f}".format(patient_isf_mean, pump_isf))

    pump_config = PumpConfig(
        basal_schedule=BasalSchedule24hr(
            t0,
            start_times=[SINGLE_SETTING_START_TIME],
            values=[BasalRate(pump_br, "U/hr")],
            duration_minutes=[SINGLE_SETTING_DURATION]
        ),
        carb_ratio_schedule=SettingSchedule24Hr(
            t0,
            "CIR",
            start_times=[SINGLE_SETTING_START_TIME],
            values=[CarbInsulinRatio(pump_cir, "g/U")],
            duration_minutes=[SINGLE_SETTING_DURATION]
        ),
        insulin_sensitivity_schedule=SettingSchedule24Hr(
            t0,
            "ISF",
            start_times=[SINGLE_SETTING_START_TIME],
            values=[InsulinSensitivityFactor(pump_isf, "mg/dL/U")],
            duration_minutes=[SINGLE_SETTING_DURATION]
        ),
        target_range_schedule=TargetRangeSchedule24hr(
            t0,
            start_times=[SINGLE_SETTING_START_TIME],
            values=[TargetRange(100, 120, "mg/dL")],
            duration_minutes=[SINGLE_SETTING_DURATION]
        ),
        carb_event_timeline=pump_carb_timeline,
        bolus_event_timeline=pump_bolus_timeline
    )

    return t0, pump_config


def compute_aace_settings_tmp(weight_kg, prepump_tdd):
    """
    AACE Pump settings calculator

    TMP: Need to put this in a more general place like metrics repo.
    """
    tdd_method1 = weight_kg * 0.5
    tdd_method2 = prepump_tdd * 0.75
    starting_pump_tdd = (tdd_method1 + tdd_method2) / 2

    basal_rate = starting_pump_tdd * 0.5 / 24
    cir = 450.0 / starting_pump_tdd
    isf = 1700.0 / starting_pump_tdd

    return basal_rate, cir, isf


def get_pump_config_from_aace_settings(random_state, patient_weight, patient_tdd, risk_level=0, t0=DATETIME_DEFAULT):

    basal_rate, cir, isf = compute_aace_settings_tmp(patient_weight, patient_tdd)

    logger.debug("Weight Kg {:.2f}. TDD {:.2f}. Pump AACE BR: {:.2f}, CIR: {:.2f}, ISF {:.2f}".format(patient_weight, patient_tdd, basal_rate, cir, isf))

    pump_carb_timeline = CarbTimeline([t0], [Carb(0.0, "g", 180)])
    pump_bolus_timeline = BolusTimeline([t0], [Bolus(0.0, "U")])

    pump_config = PumpConfig(
        basal_schedule=BasalSchedule24hr(
            t0,
            start_times=[SINGLE_SETTING_START_TIME],
            values=[BasalRate(basal_rate, "U/hr")],
            duration_minutes=[SINGLE_SETTING_DURATION]
        ),
        carb_ratio_schedule=SettingSchedule24Hr(
            t0,
            "CIR",
            start_times=[SINGLE_SETTING_START_TIME],
            values=[CarbInsulinRatio(cir, "g/U")],
            duration_minutes=[SINGLE_SETTING_DURATION]
        ),
        insulin_sensitivity_schedule=SettingSchedule24Hr(
            t0,
            "ISF",
            start_times=[SINGLE_SETTING_START_TIME],
            values=[InsulinSensitivityFactor(isf, "mg/dL/U")],
            duration_minutes=[SINGLE_SETTING_DURATION]
        ),
        target_range_schedule=TargetRangeSchedule24hr(
            t0,
            start_times=[SINGLE_SETTING_START_TIME],
            values=[TargetRange(100, 120, "mg/dL")],
            duration_minutes=[SINGLE_SETTING_DURATION]
        ),
        carb_event_timeline=pump_carb_timeline,
        bolus_event_timeline=pump_bolus_timeline
    )

    return t0, pump_config


