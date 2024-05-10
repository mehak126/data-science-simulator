import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# formatter = mdates.DateFormatter('%H:%M')
cmap = plt.cm.plasma_r

if __name__ == "__main__":
    
    start_glucose_value = 110
    basal_rate = 0.8
    cir = 8.0
    isf = 50.0    
    
    duration_hrs = 8
    
    # every 5 minutes, the basal insulin delivered is basal_rate*5/60
    
    original_bg = [start_glucose_value]*duration_hrs*12
    
    save_dir = "./simulation_results/"
    
    sims = {}
    for exercise_preset_p in np.arange(1.0, 0.01, -0.1):
        insulin_delivered = [(basal_rate/12.0)*exercise_preset_p*ii for ii in range(duration_hrs*12)]
        bg = [obg + (1-exercise_preset_p)*i*(isf/exercise_preset_p) for obg, i in zip(original_bg, insulin_delivered)]
        sim_id = f"Preset_{exercise_preset_p:.1f}"
        sims[sim_id] = {}
        sims[sim_id]['insulin'] = insulin_delivered
        sims[sim_id]['bg'] = bg
        
    fig1, ax1 = plt.subplots(dpi=150)
    fig2, ax2 = plt.subplots(dpi=150)
    
    for sim_id in sims:
        preset = float(sim_id[-3:])
        ax1.plot(sims[sim_id]['bg'], label=sim_id, c=cmap(preset))
        ax2.plot(sims[sim_id]['insulin'], label=sim_id, c=cmap(preset))
        
    ax1.legend()
    # ax1.xaxis.set_major_formatter(formatter)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Blood Glucose (mg/dL)')
    ax1.set_title(f'Starting BG {start_glucose_value}\nISF {isf}')

    plt.tight_layout()
    fig1.savefig(f'{save_dir}/estimate-bg-bg{start_glucose_value}-isf{isf}.png')

    plt.close(fig1)

    ax2.legend()
    # ax2.xaxis.set_major_formatter(formatter)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Total Insulin Delivered (U)')
    ax2.set_title(f'Starting BG {start_glucose_value}\nISF {isf}')

    plt.tight_layout()
    fig2.savefig(f'{save_dir}/estimate-insulin-bg{start_glucose_value}-isf{isf}.png')

    plt.close(fig2)
        
    
    
        
    
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    