
#import packages for data handling
import pandas as pd
import numpy as np

#import packages for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation

# Set the color palette
palette = {0: 'indianred', 1: 'steelblue'}

grad_rate = 0.4897070035943797

def bayes_error_rate(data, course_list):
    """ Calculate the Bayes error rate for a list of variables (course_list) in a dataframe (data). """
    grouped = data.groupby(course_list)
    data_grouped = pd.DataFrame({
        'COUNT(X)' : grouped.size(),
        'Pr(X)' : grouped.size() / len(data),
        'Pr(Y|X)': grouped['Y'].mean()}).reset_index()

    #add a column named "ERROR(Y|X)" which contains the minimum of Pr(Y|X) and 1-Pr(Y|X)
    data_grouped['ERROR(Y|X)'] = np.minimum(data_grouped['Pr(Y|X)'], 1 - data_grouped['Pr(Y|X)'])

    #compute the bayes error rate. This is the expected value of ERROR(Y|X) over the distribution of X
    bayes_error_rate = np.dot(data_grouped['ERROR(Y|X)'], data_grouped['Pr(X)'])
    
    return bayes_error_rate, data_grouped

def plot_course_hist_density(data,
                      course_list,
                      crse_dict,
                      savefig = False,
                      savepath = None,
                      multiple = 'dodge',
                      elements = ('bars', 'step'),
                      yscale = 'log'):
    """ Plot histogram and density plot for a list of variables (course_list) in a dataframe (data). Color according to the target variable 'Y'. """

    fig, axes = plt.subplots(len(course_list), 2, figsize=(15, len(course_list)*5), 
                            gridspec_kw={'width_ratios': [2, 1]})
        # xticks = [-9.5, -4.5, -2, -1, 0, 1, 2, 4.5, 9.5]

    for i, course in enumerate(course_list):
        positive_values_mean = data[course][data[course] > 0].mean()
        non_zero_percentage = ((data[course] != 0).sum() / len(data)) * 100
        
        # Plot histogram of raw counts with gaps between bars
        hist_data = sns.histplot(data=data, 
                                x=course, 
                                hue='Y', 
                                multiple=multiple,
                                palette=palette, 
                                ax=axes[i, 0],
                                discrete=True,
                                shrink=0.8)  # Add gap between bars
        axes[i, 0].axvline(positive_values_mean, color='black', linestyle='dashed', linewidth=1)
        # axes[i, 0].set_xticks(xticks)
        axes[i, 0].set_yscale(yscale)  # Set y-scale
        if yscale == 'log':
            axes[i, 0].set_ylim(0.1, 10000)  # Set y-limit
        
        # Add a box displaying the percentage of non-zero values
        axes[i, 0].text(0.95, 0.95, f'Non-zero: {non_zero_percentage:.1f}%', 
                        transform=axes[i, 0].transAxes, fontsize=12,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(facecolor='white', alpha=0.5))
        
        # Calculate and display the combined percentage of students in each bin
        total_counts = len(data)
        for p in hist_data.patches:
            height = p.get_height()
            if height > 0:
                percentage = f'{height / total_counts * 100:.1f}'
                axes[i, 0].annotate(percentage, 
                                    (p.get_x() + p.get_width() / 2., height), 
                                    ha='center', va='center', 
                                    xytext=(0, 9), 
                                    textcoords='offset points', 
                                    fontsize=8, color='black')
        
        # Plot probability density
        sns.histplot(data=data, 
                    x=course, 
                    hue='Y', 
                    element='poly',
                    palette=palette, 
                    ax=axes[i, 1], 
                    multiple='fill',
                    discrete=True)
        axes[i, 1].set_ylabel('Probability Density')
        axes[i, 1].axvline(positive_values_mean, color='black', linestyle='dashed', linewidth=1)
        # axes[i, 1].set_xticks(xticks)
        #remove legend from right subplot
        axes[i, 1].get_legend().remove()
        #draw a horizontal line at the graduation rate
        axes[i, 1].axhline(grad_rate, color='black', linestyle='dashed', linewidth=1)
        # Add a single title for each row, centered at the top of the row
        axes[i, 0].set_title(f'{course} : {crse_dict[course]}', loc='center', fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

    plt.tight_layout()
    if savefig:
        plt.savefig(savepath)
    plt.show()


def sem_to_year(sem):
    if sem < 0:
        return -1
    else:
        return np.ceil(sem*0.4)
    

def add_continuous_features(data, courses, math_courses, gen_courses):
    sems = [0.5*i for i in range(1, 20)]

    # Create a DataFrame for SEM columns
    sem_df = pd.concat([data[courses].apply(lambda x: x.tolist().count(i) - x.tolist().count(-i), axis=1).rename(f'SEM_{i}') for i in sems], axis=1)
    sem_math_df = pd.concat([data[math_courses].apply(lambda x: x.tolist().count(i) - x.tolist().count(-i), axis=1).rename(f'SEM_{i}_MATH') for i in sems], axis=1)
    sem_gen_df = pd.concat([data[gen_courses].apply(lambda x: x.tolist().count(i) - x.tolist().count(-i), axis=1).rename(f'SEM_{i}_GEN') for i in sems], axis=1)

    # Define columns for the cumulative sums of math performance
    data[[f'SEM_{i}_cdf' for i in sems]] = sem_df.cumsum(axis=1)
    data[[f'SEM_{i}_MATH_cdf' for i in sems]] = sem_math_df.cumsum(axis=1)
    data[[f'SEM_{i}_GEN_cdf' for i in sems]] = sem_gen_df.cumsum(axis=1)

    data['SLOPE'] = 0.0
    data['INT'] = 0.0
    data['SLOPE_MATH'] = 0.0
    data['INT_MATH'] = 0.0
    data['SLOPE_GEN'] = 0.0
    data['INT_GEN'] = 0.0

    # Loop through the rows of the DataFrame
    x = sems
    for i, row in data.iterrows():
        # y values
        y = row[[f'SEM_{j}_cdf' for j in sems]].astype(float)
        y_math = row[[f'SEM_{j}_MATH_cdf' for j in sems]].astype(float)
        y_gen = row[[f'SEM_{j}_GEN_cdf' for j in sems]].astype(float)
        # Calculate the slope and intercept
        S, I = np.polyfit(x, y, 1)
        S_math, I_math = np.polyfit(x, y_math, 1)
        S_gen, I_gen = np.polyfit(x, y_gen, 1)
        # Store the values in the original DataFrame
        data.at[i, 'SLOPE'] = S
        data.at[i, 'INT'] = I
        data.at[i, 'SLOPE_MATH'] = S_math
        data.at[i, 'INT_MATH'] = I_math
        data.at[i, 'SLOPE_GEN'] = S_gen
        data.at[i, 'INT_GEN'] = I_gen

    return data

def animate_histograms(dataframe,
                       data_series, 
                    #    titles,
                       timeperiods,
                       suptitle,  
                       unique_values,
                       interval=500,):
    """
    Animate multiple histograms
    
    Parameters:
    data_series: List of arrays to plot
    titles: List of titles for each histogram
    interval: Animation interval in milliseconds
    """
    
    # Setup the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    
    def update(frame):
        for ax in axes:
            ax.clear()
        
        # Plot histogram for raw counts
        sns.histplot(data=dataframe, 
                     x=data_series[frame], 
                     hue='Y', 
                     multiple='layer', 
                     palette=palette, 
                     ax=axes[0], 
                     discrete=True)
        axes[0].set_yscale('log')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Frequency')
        axes[0].set_xticks(unique_values)
        
        # Plot histogram for probability densities
        sns.histplot(data=dataframe, 
                     x=data_series[frame], 
                     hue='Y', 
                     multiple='fill', 
                     element='step', 
                     palette=palette, 
                     ax=axes[1], 
                     discrete=True)
        axes[1].axhline(grad_rate, color='black', linestyle='--')
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Probability Density')
        axes[1].set_xticks(unique_values)
        #draw a vertical line at the mean of the data
        axes[1].axvline(data_series[frame].mean(), color='black', linestyle='--')
        
        # Add a common title
        fig.suptitle(f"{suptitle} {timeperiods[frame]}", fontsize=16)
    
    # Create animation
    anim = FuncAnimation(fig, update,
                         frames=len(data_series),
                         interval=interval,
                         repeat=True)
    
    # Save the animation
    return anim

def generate_animations(data,sems):
    """ Generate animations for the data. """
    animate_histograms(dataframe=data,
                   data_series = [data[f'SEM_{sem}_cdf'] for sem in sems],
                #    titles = [f'SEM{sem}_cdf' for sem in sems],
                   timeperiods=sems,
                   suptitle = 'Overall course performance in semester',
                   unique_values = sorted(data['SEM_9.5_cdf'].unique())).save('../Data/Plots/animation_cdf.gif',writer='Pillow')

    df_math = data[data['TM'] == 1]
    animate_histograms(dataframe=df_math,
                    data_series = [df_math[f'SEM_{sem}_MATH_cdf'] for sem in sems],
                    timeperiods=sems,
                    suptitle = 'Math major course performance in semester',
                    unique_values = sorted(df_math['SEM_9.5_MATH_cdf'].unique())).save('../Data/Plots/animation_math.gif', writer='Pillow')

    df_nonmath = data[data['TM'] == 0]
    animate_histograms(dataframe=df_nonmath,
                    data_series = [df_nonmath[f'SEM_{sem}_GEN_cdf'] for sem in sems],
                    timeperiods=sems,
                    suptitle = 'General course performance in semester',
                    unique_values = sorted(df_nonmath['SEM_9.5_GEN_cdf'].unique())).save('../Data/Plots/animation_gen.gif', writer='Pillow')