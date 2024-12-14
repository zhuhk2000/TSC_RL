import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


def plot_aql(train_avg_queue_list, eval_avg_queue_list):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax2.plot(eval_avg_queue_list, label='eval')
    ax1.plot(train_avg_queue_list, label='train')
    ax1.set_title('train')
    ax2.set_title('eval')
    ax1.set_xlabel('Episode')
    ax2.set_xlabel('Episode')
    ax1.set_ylabel('Average queue length')
    ax2.set_ylabel('Average queue length')
    fig.savefig('train_eval_AQL.png')
    return fig

def plot_att(train_avg_travel_time_list, eval_avg_travel_time_list):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax2.plot(eval_avg_travel_time_list, label='eval')
    ax1.plot(train_avg_travel_time_list, label='train')
    ax1.set_title('train')
    ax2.set_title('eval')
    ax1.set_xlabel('Episode')
    ax2.set_xlabel('Episode')
    ax1.set_ylabel('Average travel time')
    ax2.set_ylabel('Average travel time')
    fig.savefig('train_eval_ATT.png')


def read_trip_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    travel_time = 0
    for i, tripinfo in enumerate(root.findall('tripinfo')):
        travel_time += float(tripinfo.get('duration', 0))
    else:
        return travel_time / (i + 1)
