import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def divide_income(results):

    ami = 104234
    
    high_inc = (results['income_indiv'] >= 2*ami)
    mod_inc =  (results['income_indiv'] > 0.8*ami) & (results['income_indiv'] < 2* ami)
    low_inc = (results['income_indiv'] > 0.5*ami) & (results['income_indiv'] <= 0.8* ami)
    very_low_inc = (results['income_indiv'] <= 0.5*ami)
             
    income_groups = very_low_inc, low_inc, mod_inc, high_inc
    income_labels = ['Very Low Income', 'Low Income',
             'Moderate Income', 'High Income']
    # income_colors = ['tab:purple','tab:red', 'tab:green', 'tab:orange']
    income_colors = ['#CCE5FF', '#87C3FF', '#3694FF','#0264C7']
    
    return income_groups, income_labels, income_colors

def plot_damage_ds(results):
    income_groups, labels, income_colors = divide_income(results)
    income_totals = [15384, 17631, 44624, 26696]
    
    ds_results = np.zeros((len(income_groups), 4))
    for i in range(len(income_groups)):

        for j in range(4): # damage states
            ds_results[i,j] =  len(results.loc[income_groups[i] & (results['DAMAGE'] == j+1)])/income_totals[i]*100

    df = pd.DataFrame(data = ds_results, index = labels, columns = ['Minor', 'Moderate', 'Extensive', 'Complete'])
    
    plt.figure()
    df.T.plot(kind = 'bar', color = income_colors, edgecolor = 'black', linewidth = 0.5, zorder = 3)
    plt.xlabel('Damage State')
    plt.xticks(rotation = 0)
    plt.ylabel('% of Buildings | Income Group')
    # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.legend(loc = 'upeper right')
    plt.grid(zorder = 0, alpha = 0.3)
    plt.show()

def plot_cons_fin_ds(results):
    income_groups, income_labels, income_colors = divide_income(results)

    # Compare cons start and financing time
    plot_data = []
    for i in range(len(income_groups)):
        data = results.loc[income_groups[i] & (results['cons_start'].notnull()) & (results['DAMAGE'] >= 3), ['cons_start', 'total_time']]
        data['cons_start'] = data['cons_start']/365
        data['total_time'] = data['total_time']/365
        plot_data.append(data)
    
    bins = np.histogram(np.hstack((plot_data[0]['cons_start'],plot_data[0]['total_time'],
                                  plot_data[1]['cons_start'],plot_data[1]['total_time'],
                                  plot_data[2]['cons_start'],plot_data[2]['total_time'],
                                  plot_data[3]['cons_start'],plot_data[3]['total_time'])), bins=30)[1]
    
    fig, axes = plt.subplots(figsize = (10, 6), nrows=2, ncols=2)
    for i in range(len(income_groups)):
        plt.subplot(2,2,i+1)
        
        plt.hist(plot_data[i]['cons_start'], bins = bins, alpha = 0.5, 
             label = 'Construction Start', color = income_colors[i])
        plt.hist(plot_data[i]['total_time'],bins = bins, alpha = 0.5, 
                         label = 'Financing Received', color = 'tab:grey')   
                
        cons_mean = np.mean(plot_data[i]['cons_start'])
        fin_mean = np.mean(plot_data[i]['total_time'])
        
        ymin, ymax = plt.gca().get_ylim()
        plt.vlines(cons_mean, 0, ymax, color = 'tab:red', linestyles = 'dashed', label = 'Construction Start Mean')
        plt.vlines(fin_mean, 0, ymax, linestyles = 'dashed', label = 'Financing Received Mean')
                
        plt.xlabel('Years')
        plt.ylabel('Number of Buildings')
        
        plt.ylim([0,ymax])
        plt.xlim([-0.5, 8])
        plt.grid(alpha = 0.3)
        plt.title(income_labels[i])
        plt.legend()

    fig.tight_layout()
    plt.show()

def plot_recovery_curve(results):
    income_groups, labels, income_colors = divide_income(results)

    total_building = len(results)
    finish_time = results['cons_finish'].values
    finish_time = finish_time[~np.isnan(finish_time)]
    
    plt.figure()
    for i in range(len(income_groups)):
        total_group = sum(income_groups[i])
        finish_group = results.loc[income_groups[i], 'cons_finish'].values
        finish_group = finish_group[~np.isnan(finish_group)]
        y_group = np.arange(len(finish_group))/float(total_group-1)
        
        plt.step(np.concatenate([np.sort(finish_group), [np.sort(finish_time)[-1]]])/365,
             np.concatenate([y_group, [y_group[-1]]])*100, label = labels[i], color = income_colors[i])

    plt.ylim([-5, 110])
    plt.xlim([-0.5, 8])
    plt.xlabel('Years')
    plt.ylabel('Percent Completion')
    # plt.title('Recovery Curve')
    plt.legend()
    plt.grid(alpha = 0.3)
    plt.show()

def plot_recovery_bar(results):

    income_groups, income_labels, income_colors = divide_income(results)
    
    ds_results = np.zeros((len(income_groups), 4))
    for i in range(len(income_groups)):

        for j in range(4): # damage states
            total_group = len(results.loc[income_groups[i] & (results['DAMAGE'] == j+1)])
            finish_group = results.loc[income_groups[i] & (results['DAMAGE'] == j+1), 'cons_finish'].values
            finish_group = len(finish_group[~np.isnan(finish_group)])

            ds_results[i,j] = finish_group/total_group*100
            
    df = pd.DataFrame(data = ds_results, index = income_labels, columns = ['Minor', 'Moderate', 'Extensive', 'Complete'])

    plt.figure()
    df.T.plot(kind = 'bar', color = income_colors, edgecolor = 'black', linewidth = 0.5, zorder = 3)
    plt.xlabel('Damage State')
    plt.xticks(rotation = 0)
    plt.ylabel('% of buildings fully reconstructed')
    plt.legend().remove()
    plt.grid(zorder = 0, alpha = 0.3)

    # Shrink current axis's height by 10% on the bottom
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    plt.show()