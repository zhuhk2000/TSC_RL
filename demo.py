import xml.etree.ElementTree as ET
import os



file_path = './data/output/summary_rl.xml'

tree = ET.parse(file_path)
root = tree.getroot()

total_halting_vehicles = 0

for step in root.findall('step'):
    halting_vehicles = int(step.get('halting', '0'))
    total_halting_vehicles += halting_vehicles

print(f"Total halting vehicles during the simulation: {total_halting_vehicles}")
