
#import packages for data handling
import pandas as pd
import numpy as np

#import packages for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set the color palette
palette = {0: 'indianred', 1: 'steelblue'}

def plot_course_hist_density(data,
                      course_list,
                      multiple = 'dodge',
                      elements = ('bars', 'step'),
                      yscale = 'log'):
    """ Plot histogram and density plot for a list of variables (course_list) in a dataframe (data). Color according to the target variable 'Y'. """

    fig, axes = plt.subplots(len(course_list), 2, figsize=(20, len(course_list)*4))
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
                                    fontsize=10, color='black')
        
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
        
        # Add a single title for each row, centered at the top of the row
        axes[i, 0].set_title(f'{course}', loc='center', fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

    plt.tight_layout()
    # plt.savefig('gen_histograms_density.png')
    plt.show()
