from flask import Flask, jsonify, render_template, request
import pickle
import numpy as np

# model = pickle.load(open('model.pkl', 'rb'))
# print("Model loaded successfully!")
model2 = pickle.load(open('./reg_model.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def predict():
 return render_template('index.html')     
#     symptom_dictionary = {
#         'symptom_index': {'Itching': 0, 'Skin Rash': 1, 'Nodal Skin Eruptions': 2, 'Continuous Sneezing': 3,
#                           'Shivering': 4, 'Chills': 5, 'Joint Pain': 6, 'Stomach Pain': 7, 'Acidity': 8,
#                           'Ulcers On Tongue': 9, 'Muscle Wasting': 10, 'Vomiting': 11,
#                           'Burning Micturition': 12, 'Spotting  urination': 13, 'Fatigue': 14,
#                           'Weight Gain': 15, 'Anxiety': 16, 'Cold Hands And Feets': 17, 'Mood Swings': 18,
#                           'Weight Loss': 19, 'Restlessness': 20, 'Lethargy': 21, 'Patches In Throat': 22,
#                           'Irregular Sugar Level': 23, 'Cough': 24, 'High Fever': 25, 'Sunken Eyes': 26,
#                           'Breathlessness': 27, 'Sweating': 28, 'Dehydration': 29, 'Indigestion': 30,
#                           'Headache': 31, 'Yellowish Skin': 32, 'Dark Urine': 33, 'Nausea': 34,
#                           'Loss Of Appetite': 35, 'Pain Behind The Eyes': 36, 'Back Pain': 37,
#                           'Constipation': 38, 'Abdominal Pain': 39, 'Diarrhoea': 40, 'Mild Fever': 41,
#                           'Yellow Urine': 42, 'Yellowing Of Eyes': 43, 'Acute Liver Failure': 44,
#                           'Fluid Overload': 45, 'Swelling Of Stomach': 46, 'Swelled Lymph Nodes': 47,
#                           'Malaise': 48, 'Blurred And Distorted Vision': 49, 'Phlegm': 50,
#                           'Throat Irritation': 51, 'Redness Of Eyes': 52, 'Sinus Pressure': 53,
#                           'Runny Nose': 54, 'Congestion': 55, 'Chest Pain': 56, 'Weakness In Limbs': 57,
#                           'Fast Heart Rate': 58, 'Pain During Bowel Movements': 59, 'Pain In Anal Region': 60,
#                           'Bloody Stool': 61, 'Irritation In Anus': 62, 'Neck Pain': 63, 'Dizziness': 64,
#                           'Cramps': 65, 'Bruising': 66, 'Obesity': 67, 'Swollen Legs': 68,
#                           'Swollen Blood Vessels': 69, 'Puffy Face And Eyes': 70, 'Enlarged Thyroid': 71,
#                           'Brittle Nails': 72, 'Swollen Extremeties': 73, 'Excessive Hunger': 74,
#                           'Extra Marital Contacts': 75, 'Drying And Tingling Lips': 76, 'Slurred Speech': 77,
#                           'Knee Pain': 78, 'Hip Joint Pain': 79, 'Muscle Weakness': 80, 'Stiff Neck': 81,
#                           'Swelling Joints': 82, 'Movement Stiffness': 83, 'Spinning Movements': 84,
#                           'Loss Of Balance': 85, 'Unsteadiness': 86, 'Weakness Of One Body Side': 87,
#                           'Loss Of Smell': 88, 'Bladder Discomfort': 89, 'Foul Smell Of urine': 90,
#                           'Continuous Feel Of Urine': 91, 'Passage Of Gases': 92, 'Internal Itching': 93,
#                           'Toxic Look (typhos)': 94, 'Depression': 95, 'Irritability': 96, 'Muscle Pain': 97,
#                           'Altered Sensorium': 98, 'Red Spots Over Body': 99, 'Belly Pain': 100,
#                           'Abnormal Menstruation': 101, 'Dischromic  Patches': 102, 'Watering From Eyes': 103,
#                           'Increased Appetite': 104, 'Polyuria': 105, 'Family History': 106,
#                           'Mucoid Sputum': 107, 'Rusty Sputum': 108, 'Lack Of Concentration': 109,
#                           'Visual Disturbances': 110, 'Receiving Blood Transfusion': 111,
#                           'Receiving Unsterile Injections': 112, 'Coma': 113, 'Stomach Bleeding': 114,
#                           'Distention Of Abdomen': 115, 'History Of Alcohol Consumption': 116,
#                           'Fluid Overload.1': 117, 'Blood In Sputum': 118, 'Prominent Veins On Calf': 119,
#                           'Palpitations': 120, 'Painful Walking': 121, 'Pus Filled Pimples': 122,
#                           'Blackheads': 123, 'Scurring': 124, 'Skin Peeling': 125, 'Silver Like Dusting': 126,
#                           'Small Dents In Nails': 127, 'Inflammatory Nails': 128, 'Blister': 129,
#                           'Red Sore Around Nose': 130, 'Yellow Crust Ooze': 131},
#         'predictions_classes': ['(vertigo) Paroymsal  Positional Vertigo', 'AIDS', 'Acne',
#                                 'Alcoholic hepatitis', 'Allergy', 'Arthritis', 'Bronchial Asthma',
#                                 'Cervical spondylosis', 'Chicken pox', 'Chronic cholestasis',
#                                 'Common Cold', 'Dengue', 'Diabetes ',
#                                 'Dimorphic hemmorhoids(piles)', 'Drug Reaction',
#                                 'Fungal infection', 'GERD', 'Gastroenteritis', 'Heart attack',
#                                 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
#                                 'Hypertension ', 'Hyperthyroidism', 'Hypoglycemia',
#                                 'Hypothyroidism', 'Impetigo', 'Jaundice', 'Malaria', 'Migraine',
#                                 'Osteoarthristis', 'Paralysis (brain hemorrhage)',
#                                 'Peptic ulcer diseae', 'Pneumonia', 'Psoriasis', 'Tuberculosis',
#                                 'Typhoid', 'Urinary tract infection', 'Varicose veins',
#                                 'hepatitis A']}
#     disease_with_explanation = {
#         'Drug Reaction': 'A drug allergy is the reaction of the immune system to a medicine. Any medicine — nonprescription, prescription or herbal — can provoke a drug allergy. However, a drug allergy is more likely with certain medicines.',
#         'Malaria': 'Malaria is a disease caused by a parasite. The parasite is spread to humans through the bites of infected mosquitoes. People who have malaria usually feel very sick with a high fever and shaking chills. While the disease is uncommon in temperate climates, malaria is still common in tropical and subtropical countries. Each year nearly 290 million people are infected with malaria, and more than 400,000 people die of the disease.',
#         'Allergy': "Allergies occur when your immune system reacts to a foreign substance — such as pollen, bee venom or pet dander — or a food that doesn't cause a reaction in most people.\nThe severity of allergies varies from person to person and can range from minor irritation to anaphylaxis — a potentially life-threatening emergency. While most allergies can't be cured, treatments can help relieve your allergy symptoms.",
#         'Hypothyroidism': "Hyperthyroidism happens when the thyroid gland makes too much thyroid hormone. This condition also is called overactive thyroid. Hyperthyroidism speeds up the body's metabolism. That can cause many symptoms, such as weight loss, hand tremors, and rapid or irregular heartbeat.",
#         'Psoriasis': 'Psoriasis is a skin disease that causes a rash with itchy, scaly patches, most commonly on the knees, elbows, trunk and scalp. Psoriasis is a common, long-term (chronic) disease with no cure. It can be painful, interfere with sleep and make it hard to concentrate. The condition tends to go through cycles, flaring for a few weeks or months, then subsiding for a while. Common triggers in people with a genetic predisposition to psoriasis include infections, cuts or burns, and certain medications.',
#         'GERD': 'Gastroesophageal reflux disease (GERD) occurs when stomach acid repeatedly flows back into the tube connecting your mouth and stomach (esophagus). This backwash (acid reflux) can irritate the lining of your esophagus.\nMany people experience acid reflux from time to time. However, when acid reflux happens repeatedly over time, it can cause GERD. ',
#         'Chronic cholestasis': 'Cholestasis is a liver disease. It occurs when the flow of bile from your liver is reduced or blocked. Bile is fluid produced by your liver that aids in the digestion of food, especially fats. When bile flow is altered, it can lead to a buildup of bilirubin. Bilirubin is a pigment produced by your liver and excreted from your body via bile.',
#         'hepatitis A': "Hepatitis A is a highly contagious liver infection caused by the hepatitis A virus. The virus is one of several types of hepatitis viruses that cause liver inflammation and affect your liver's ability to function.",
#         'Osteoarthristis': 'Osteoarthritis is the most common form of arthritis, affecting millions of people worldwide. It occurs when the protective cartilage that cushions the ends of the bones wears down over time. Although osteoarthritis can damage any joint, the disorder most commonly affects joints in your hands, knees, hips and spine.',
#         '(vertigo) Paroymsal  Positional Vertigo': "Benign paroxysmal positional vertigo (BPPV) is one of the most common causes of vertigo — the sudden sensation that you're spinning or that the inside of your head is spinning. BPPV causes brief episodes of mild to intense dizziness. It is usually triggered by specific changes in your head's position. This might occur when you tip your head up or down, when you lie down, or when you turn over or sit up in bed.",
#         'Hypoglycemia': "Hypoglycemia is a condition in which your blood sugar (glucose) level is lower than the standard range. Glucose is your body's main energy source. Hypoglycemia is often related to diabetes treatment. But other drugs and a variety of conditions — many rare — can cause low blood sugar in people who don't have diabetes.",
#         'Acne': 'Acne is a skin condition that occurs when your hair follicles become plugged with oil and dead skin cells. It causes whiteheads, blackheads or pimples. Acne is most common among teenagers, though it affects people of all ages. Depending on its severity, acne can cause emotional distress and scar the skin. The earlier you start treatment, the lower your risk of such problems.',
#         'Diabetes': "Diabetes mellitus refers to a group of diseases that affect how the body uses blood sugar (glucose). Glucose is an important source of energy for the cells that make up the muscles and tissues. It's also the brain's main source of fuel. The main cause of diabetes varies by type. But no matter what type of diabetes you have, it can lead to excess sugar in the blood. Too much sugar in the blood can lead to serious health problems.",
#         'Impetigo': 'Impetigo (im-puh-TIE-go) is a common and highly contagious skin infection that mainly affects infants and young children. It usually appears as reddish sores on the face, especially around the nose and mouth and on the hands and feet. Over about a week, the sores burst and develop honey-colored crusts.',
#         'Hypertension': 'Pulmonary hypertension is a type of high blood pressure that affects the arteries in the lungs and the right side of the heart. In one form of pulmonary hypertension, called pulmonary arterial hypertension (PAH), blood vessels in the lungs are narrowed, blocked or destroyed. The damage slows blood flow through the lungs, and blood pressure in the lung arteries rises. The heart must work harder to pump blood through the lungs. The extra effort eventually causes the heart muscle to become weak and fail.',
#         'Peptic ulcer diseae': 'Peptic ulcers are open sores that develop on the inside lining of your stomach and the upper portion of your small intestine. The most common symptom of a peptic ulcer is stomach pain.',
#         'Dimorphic hemorrhoids(piles)': 'Hemorrhoids (HEM-uh-roids), also called piles, are swollen veins in your anus and lower rectum, similar to varicose veins. Hemorrhoids can develop inside the rectum (internal hemorrhoids) or under the skin around the anus (external hemorrhoids).',
#         'Common Cold': "The common cold is a viral infection of your nose and throat (upper respiratory tract). It's usually harmless, although it might not feel that way. Many types of viruses can cause a common cold. Healthy adults can expect to have two or three colds each year. Infants and young children may have even more frequent colds.",
#         'Chicken pox': "Chickenpox is an infection caused by the varicella-zoster virus. It causes an itchy rash with small, fluid-filled blisters. Chickenpox is highly contagious to people who haven't had the disease or been vaccinated against it. Today, a vaccine is available that protects children against chickenpox. Routine vaccination is recommended by the U.S. Centers for Disease Control and Prevention (CDC).",
#         'Cervical spondylosis': 'Cervical spondylosis is a general term for age-related wear and tear affecting the spinal disks in your neck. As the disks dehydrate and shrink, signs of osteoarthritis develop, including bony projections along the edges of bones (bone spurs).',
#         'Hyperthyroidism': "Hyperthyroidism happens when the thyroid gland makes too much thyroid hormone. This condition also is called overactive thyroid. Hyperthyroidism speeds up the body's metabolism. That can cause many symptoms, such as weight loss, hand tremors, and rapid or irregular heartbeat.",
#         'Urinary tract infection': 'A urinary tract infection (UTI) is an infection in any part of the urinary system. The urinary system includes the kidneys, ureters, bladder and urethra. Most infections involve the lower urinary tract — the bladder and the urethra.',
#         'Varicose veins': "Varicose veins are twisted, enlarged veins. Any vein that is close to the skin's surface (superficial) can become varicosed. Varicose veins most commonly affect the veins in the legs. That's because standing and walking increase the pressure in the veins of the lower body.",
#         'AIDS': "Acquired immunodeficiency syndrome (AIDS) is a chronic, potentially life-threatening condition caused by the human immunodeficiency virus (HIV). By damaging your immune system, HIV interferes with your body's ability to fight infection and disease.",
#         'Paralysis (brain hemorrhage)': 'A subarachnoid hemorrhage is bleeding in the space between the brain and the surrounding membrane (subarachnoid space). The primary symptom is a sudden, severe headache. Some people describe it as the worst headache they have ever felt. ',
#         'Typhoid': 'Typhoid fever, also called enteric fever, is caused by salmonella bacteria. Typhoid fever is rare in places where few people carry the bacteria. It also is rare where water is treated to kill germs and where human waste disposal is managed. One example of where typhoid fever is rare is the United States. Places with the highest number of cases or with regular outbreaks are in Africa and South Asia. It is a serious health threat, especially for children, in places where it is more common.',
#         'Hepatitis B': 'Hepatitis B is a serious liver infection caused by the hepatitis B virus (HBV). For most people, hepatitis B is short term, also called acute, and lasts less than six months. But for others, the infection becomes chronic, meaning it lasts more than six months. Having chronic hepatitis B increases your risk of developing liver failure, liver cancer or cirrhosis — a condition that permanently scars the liver.',
#         'Fungal infection': "A fungus that invades the tissue can cause a disease that's confined to the skin, spreads into tissue, bones and organs or affects the whole body. Symptoms depend on the area affected, but can include skin rash or vaginal infection resulting in abnormal discharge. ",
#         'Hepatitis C': 'Hepatitis C is a viral infection that causes liver inflammation, sometimes leading to serious liver damage. The hepatitis C virus (HCV) spreads through contaminated blood.',
#         'Migraine': "A migraine is a headache that can cause severe throbbing pain or a pulsing sensation, usually on one side of the head. It's often accompanied by nausea, vomiting, and extreme sensitivity to light and sound. Migraine attacks can last for hours to days, and the pain can be so severe that it interferes with your daily activities.",
#         'Bronchial Asthma': 'Bronchial  Asthma is a condition in which your airways narrow and swell and may produce extra mucus. This can make breathing difficult and trigger coughing, a whistling sound (wheezing) when you breathe out and shortness of breath.',
#         'Alcoholic hepatitis': 'Alcoholic hepatitis is inflammation of the liver caused by drinking alcohol. Alcoholic hepatitis is most likely to occur in people who drink heavily over many years. However, the relationship between drinking and alcoholic hepatitis is complex. Not all heavy drinkers develop alcoholic hepatitis, and the disease can occur in people who drink only moderately.',
#         'Jaundice': "Jaundice may occur if the liver can't efficiently process red blood cells as they break down. It's normal in healthy newborns and usually clears on its own. At other ages, it may signal infection or liver disease.",
#         'Hepatitis E': 'Hepatitis E is an inflammation of the liver caused by infection with the hepatitis E virus (HEV). The hepatitis E virus is mainly transmitted through drinking water contaminated with faecal matter.',
#         'Dengue': 'Dengue (DENG-gey) fever is a mosquito-borne illness that occurs in tropical and subtropical areas of the world. Mild dengue fever causes a high fever and flu-like symptoms. The severe form of dengue fever, also called dengue hemorrhagic fever, can cause serious bleeding, a sudden drop in blood pressure (shock) and death.',
#         'Hepatitis D': 'Hepatitis D is a liver infection you can get if you have hepatitis B. It can cause serious symptoms that can lead to lifelong liver damage and even death. It’s sometimes called hepatitis delta virus (HDV) or delta hepatitis.',
#         'Heart attack': 'A heart attack occurs when the flow of blood to the heart is severely reduced or blocked. The blockage is usually due to a buildup of fat, cholesterol and other substances in the heart (coronary) arteries. The fatty, cholesterol-containing deposits are called plaques. The process of plaque buildup is called atherosclerosis. Sometimes, a plaque can rupture and form a clot that blocks blood flow. A lack of blood flow can damage or destroy part of the heart muscle.',
#         'Pneumonia': 'Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus (purulent material), causing cough with phlegm or pus, fever, chills, and difficulty breathing. A variety of organisms, including bacteria, viruses and fungi, can cause pneumonia. Pneumonia can range in seriousness from mild to life-threatening. It is most serious for infants and young children, people older than age 65, and people with health problems or weakened immune systems.',
#         'Arthritis': 'Arthritis is the swelling and tenderness of one or more joints. The main symptoms of arthritis are joint pain and stiffness, which typically worsen with age. The most common types of arthritis are osteoarthritis and rheumatoid arthritis.',
#         'Gastroenteritis': "Viral gastroenteritis is an intestinal infection that includes signs and symptoms such as watery diarrhea, stomach cramps, nausea or vomiting, and sometimes fever. The most common way to develop viral gastroenteritis — often called stomach flu — is through contact with an infected person or by consuming contaminated food or water. If you're otherwise healthy, you'll likely recover without complications. But for infants, older adults and people with compromised immune systems, viral gastroenteritis can be deadly.",
#         'Tuberculosis': 'Tuberculosis (TB) is a potentially serious infectious disease that mainly affects the lungs. The bacteria that cause tuberculosis are spread from person to person through tiny droplets released into the air via coughs and sneezes.'}
#     disease_with_symptoms = {
#         'Drug Reaction': '<span>Skin rash</span><span>Hives</span><span>Itching</span><span>Fever</span><span>Swelling</span><span>Shortness of breath</span><span>Wheezing</span><span>Runny nose</span><span>Itchy/watery eyes</span><span>Runny nose</span>',
#         'Malaria': '<span>Fever</span> <span>Chills</span> <span>Discomfort</span> <span>Headache</span> <span>Nausea</span> <span>Vomiting</span> <span>Diarrhea</span> <span>Abdominal pain</span> <span>Muscle pain</span> <span>Joint pain</span> <span>Fatigue</span> <span>Rapid breathing</span> <span>Rapid heart rate</span>',
#         'Allergy': '<span>Sneezing</span> <span>Itching nose</span> <span>Watery eyes</span> <span>Stuffy nose</span> <span>Conjunctivitis</span>',
#         'Hypothyroidism': '<span>Tiredness</span> <span>Sensitivity to cold</span> <span>Constipation</span> <span>Dry skin</span> <span>Weight gain</span> <span>Puffy face</span> <span>Hoarse voice</span> <span>Coarse hair and skin</span> <span>Muscle weakness</span> <span>Muscle aches</span>',
#         'Psoriasis': '<span>Patchy rash</span> <span>Dandruff</span> <span>Scaling spots</span> <span>Dry cracked skin</span> <span>Itching</span> <span>Burning or soreness</span> <span>Cyclic rashes</span>',
#         'GERD': '<span>Heartburn</span><span>Backwash</span><span>Abdominal pain</span><span>Chest Pain</span><span>Dysphagia</span><span>Globus</span>',
#         'Chronic cholestasis': '<span>Jaundice</span> <span>Dark urine</span> <span>Light colored stool</span> <span>Abdomen pain</span> <span>Fatigue</span> <span>Nausea</span> <span>Excessive itching</span>',
#         'hepatitis A': '<span>Tiredness</span><span>Weakness</span><span>Nausea</span><span>Vomiting</span><span>Diarrhea</span><span>Abdominal pain</span><span>Grey stool</span><span>Loss of appetite</span><span>Mild fever</span><span>Dark urine</span><span>Joint pain</span><span>Yellow skin</span>',
#         'Osteoarthristis': '<span>Joint pain</span> <span>Stiffness</span> <span>Loss of flexibility</span> <span>Grating sensation</span> <span>Bone spurs</span> <span>Inflammation</span> <span>swelling</span>',
#         '(vertigo) Paroymsal  Positional Vertigo': '<span>Dizziness</span> <span>Vertigo</span> <span>Unsteadiness</span> <span>Nausea</span> <span>Vomiting</span>',
#         'Hypoglycemia': '<span>Pale face</span> <span>Shivering</span> <span>Sweating</span> <span>Headache</span> <span>Hunger</span> <span>Nausea</span> <span>Rapid heartbeat</span> <span>Fatigue</span> <span>Anxiety</span> <span>Dizziness</span> <span>Tingling</span>',
#         'Acne': '<span>Whiteheads</span><span>Blackheads</span><span>Papules</span><span>Pimples</span><span>Nodules</span><span>Cystic lesions</span>',
#         'Diabetes': '<span>Feelingthirsty</span><span>Urinatingoften</span><span>WeightLoss</span><span>ketonesintheurine</span><span>Exhaustion</span><span>Moodswings</span><span>Blurryvision</span><span>SlowHealing</span>',
#         'Impetigo': '<span>Reddish sores</span><span>larger blisters</span><span>Ecthyma</span>',
#         'Hypertension': '<span>Blue lips & skin</span><span>Chest Pain</span><span>Dizziness</span><span>Palpitations</span><span>Fatigue</span><span>Shortness of breath</span><span>Edema</span>',
#         'Peptic ulcer diseae': '<span>Stomach pain</span><span>Bloating</span> <span>Belching</span> <span>Intolerance to fatty foods</span> <span>Heartburn</span> <span>Nausea</span>',
#         'Dimorphic hemorrhoids(piles)': '<span>Itching anus</span> <span>irritation anus</span> <span>Pain or discomfort</span> <span>Swelling around anus</span> <span>Bleeding</span>',
#         'Common Cold': '<span>Runny nose</span><span>Sore throat</span><span>Cough</span><span>Congestion</span><span>Headache</span><span>Body aches</span><span>Sneezing</span><span>Mild fever</span>',
#         'Chicken pox': '<span>Itchy blister</span><span>Rash</span><span>Fever</span><span>Loss of appetite</span><span>Headache</span><span>Tiredness</span>',
#         'Cervical spondylosis': '<span>Pain & stiffness in neck</span> <span>Tingling</span> <span>Weak arms & legs</span> <span>Uncoordinated Movement</span> <span>Bladder dysfunction</span>',
#         'Hyperthyroidism': '<span>Weight loss</span> <span>Tachycardia</span> <span>Heart palpitations</span> <span>Increased hunger</span> <span>Anxiety</span> <span>Tremor</span> <span>Sweating</span> <span>Exhaustion</span> <span>Muscle weakness</span> <span>Sleep problems</span> <span>Moist skin</span> <span>Brittle hair</span>',
#         'Urinary tract infection': '<span>Strong urge to urinate</span> <span>Burning while urination</span> <span>Frequent urination</span> <span>Cloudy urine</span> <span>Foul urine smell</span> <span>Pelvic pain</span>',
#         'Varicose veins': '<span>Blue veins</span> <span>Bulging veins</span> <span>Achy legs</span> <span>Muscle cramp</span> <span>Swelling</span> <span>Skin color change</span>',
#         'AIDS': '<span>Fever</span><span>Headache</span><span>Muscle aches</span><span>Joint pain</span><span>Rash</span><span>Sore throat</span><span>Severe mouth sores</span><span>Swollen lymph glands</span>',
#         'Paralysis (brain hemorrhage)': '<span>Nausea</span> <span>Vomiting</span> <span>Stiff neck</span> <span>Problems with vision</span> <span>Loss of consciousness</span>',
#         'Typhoid': '<span>High fever</span> <span>Chills</span> <span>Headache</span> <span>Weakness and fatigue</span> <span>Muscle aches</span> <span>Stomach pain</span> <span>Diarrhea or Constipation</span> <span>Rash</span>',
#         'Hepatitis B': '<span>Abdominal pain</span> <span>Dark urine</span> <span>Fever</span> <span>Joint pain</span> <span>Loss of appetite</span> <span>Nausea and vomiting</span> <span>Weakness and fatigue</span> <span>Jaundice</span>',
#         'Fungal infection': '<span>Dark skin</span> <span>Loss of colour</span> <span>Peeling</span> <span>Rashes</span> <span>Small bump</span> <span>Deformed toenail</span> <span>Itching</span>',
#         'Hepatitis C': '<span>Easy Bleeding</span> <span>Easy Bruising</span> <span>Fatigue</span> <span>Poor appetite</span> <span>jaundice</span> <span>Dark urine</span> <span>Itchy skin</span> <span>Ascites</span> <span>Leg Swelling</span> <span>Weight loss</span> <span>Confusion & drowsiness</span> <span>Spider blood vessels</span>',
#         'Migraine': '<span>Constipation</span> <span>Mood swings</span> <span>Food cravings</span> <span>Neck stiffness</span> <span>Increased urination</span> <span>Fluid retention</span> <span>Frequent yawning</span>',
#         'Bronchial Asthma': '<span>Shortness of breath</span> <span>Chest tightness or pain</span> <span>Wheezing when exhaling</span> <span>Coughing</span> <span>Wheezing</span>',
#         'Alcoholic hepatitis': '<span>Loss of appetite</span> <span>Nausea</span> <span>Vomiting</span> <span>Abdominal tenderness</span> <span>Mild fever</span> <span>Fatigue</span> <span>Weakness</span>',
#         'Jaundice': '<span>Yellow skin</span> <span>Yellow eyes</span>',
#         'Hepatitis E': '<span>Mild fever</span> <span>Fatigue</span> <span>Nausea</span> <span>Throwing up</span> <span>Stomach pain</span> <span>Dark urine</span> <span>Light stool</span> <span>Skin rash</span> <span>Itching</span> <span>Joint pain</span> <span>Jaundice</span>',
#         'Dengue': '<span>High fever</span> <span>Headache</span> <span>Muscle pain</span> <span>Joint pain</span> <span>Nausea</span> <span>Vomiting</span> <span>Pain behind the eyes</span> <span>Swollen glands</span> <span>Rash</span>',
#         'Hepatitis D': '<span>Jaundice</span> <span>Nausea</span> <span>Stomach Pain</span> <span>Fatigue</span> <span>No hunger</span> <span>Joint pain</span> <span>Dark urine</span> <span>Light stool</span>',
#         'Heart attack': '<span>Chest pain</span> <span>Shoulder pain</span> <span>Neck pain</span> <span>Stomach pain</span> <span>Cold sweat</span> <span>Fatigue</span> <span>Heartburn</span> <span>Indigestion</span> <span>Lightheadedness</span> <span>Nausea</span> <span>Shortness of breath</span>',
#         'Pneumonia': '<span>Chest pain</span> <span>Cough</span> <span>Confusion</span> <span>Fatigue</span> <span>Fever</span> <span>Sweating</span> <span>Shaking chills</span> <span>Low body temp</span> <span>Nausea</span> <span>Vomiting</span> <span>Diarrhea</span> <span>Shortness of breath</span>',
#         'Arthritis': '<span>Joint Pain</span> <span>Stiffness</span> <span>Swelling</span> <span>Redness</span> <span>Decreased range of motion</span>',
#         'Gastroenteritis': '<span>Watery diarrhea</span> <span>Nausea</span> <span>Vomiting</span> <span>Stomach cramps</span> <span>Muscle aches</span> <span>Headache</span> <span>Mild fever</span>',
#         'Tuberculosis': '<span>Coughing</span> <span>Chest pain</span> <span>Weight loss</span> <span>Fatigue</span> <span>Fever</span> <span>Night sweats</span> <span>Chills</span> <span>Loss of appetite</span>'}
#     disease_with_treatment = {
#         'Drug Reaction': 'The following interventions may be used to treat an allergic reaction to a drug: Withdrawal of the drug, Antihistamines, Corticosteroids, Treatment of anaphylaxis',
#         'Malaria': 'Malaria is treated with prescription drugs to kill the parasite. The types of drugs and the length of treatment will vary, depending on: malaria parasite, severity, age and pregnancy. The most common antimalarial drugs include: Chloroquine phosphate and Artemisinin-based combination therapies (ACTs).',
#         'Allergy': 'Nasal steroid sprays are generally the most effective medication for people with allergic rhinitis symptoms. Antihistamines block some of the effects of histamine and may offer additional benefits. Immunotherapy helps create a tolerance to allergens and can improve many of the symptoms related to inhalant allergy exposure.',
#         'Hypothyroidism': 'Treatment for hypothyroidism usually includes taking the thyroid hormone medicine levothyroxine (Levo-T, Synthroid, others) every day. This medicine is taken by mouth. It returns hormone levels to a healthy range, eliminating symptoms of hypothyroidism.',
#         'Psoriasis': 'Psoriasis treatments aim to stop skin cells from growing so quickly and to remove scales. Options include creams and ointments (topical therapy), light therapy (phototherapy), and oral or injected medications.',
#         'GERD': "Antacids that neutralize stomach acid. Antacids containing calcium carbonate, such as Mylanta, Rolaids and Tums, may provide quick relief. But antacids alone won't heal an inflamed esophagus damaged by stomach acid. Overuse of some antacids can cause side effects, such as diarrhea or sometimes kidney problems, Medications to reduce acid production — known as histamine (H-2) blockers — include cimetidine (Tagamet HB), famotidine (Pepcid AC) and nizatidine (Axid AR). H-2 blockers don't act as quickly as antacids, but they provide longer relief and may decrease acid production from the stomach for up to 12 hours. Stronger versions are available by prescription.",
#         'Chronic cholestasis': 'The first step to treating cholestasis is to treat the underlying cause. For example, if it’s determined that medication is causing the condition, your doctor may recommend a different drug. If an obstruction like gallstones or a tumor is causing the backup of bile, your doctor may recommend surgery. In most cases, obstetric cholestasis resolves after delivery. Women who develop obstetric cholestasis should be monitored post-pregnancy.',
#         'hepatitis A': "Antiviral medications. Several antiviral medicines — including entecavir (Baraclude), tenofovir (Viread), lamivudine (Epivir), adefovir (Hepsera) and telbivudine — can help fight the virus and slow its ability to damage your liver. These drugs are taken by mouth. Interferon injections. Interferon alfa-2b (Intron A) is a man-made version of a substance produced by the body to fight infection. It's used mainly for young people with hepatitis B who wish to avoid long-term treatment or women who might want to get pregnant within a few years, after completing a finite course of therapy. ",
#         'Osteoarthristis': "Osteoarthritis can't be reversed, but treatments can reduce pain and help you move better. Medication, physiotherapy and sometimes surgery can help reduce pain and maintain joint movement.",
#         '(vertigo) Paroymsal  Positional Vertigo': 'Paroxysmal positional vertigo may go away on its own within a few weeks or months. But, to help relieve BPPV sooner, your doctor, audiologist or physical therapist may treat you with a series of movements known as the canalith repositioning procedure.',
#         'Hypoglycemia': "Eat or drink 15 to 20 grams of fast-acting carbohydrates. These are sugary foods or drinks without protein or fat that are easily converted to sugar in the body. Try glucose tablets or gel, fruit juice, regular (not diet) soda, honey, or sugary candy. Recheck blood sugar levels 15 minutes after treatment. If blood sugar levels are still under 70 mg/dL (3.9 mmol/L), eat or drink another 15 to 20 grams of fast-acting carbohydrate, and recheck your blood sugar level again in 15 minutes. Have a snack or meal. Once your blood sugar is back in the standard range, eating a healthy snack or meal can help prevent another drop in blood sugar and replenish your body's glycogen stores.",
#         'Acne': "If you've tried over-the-counter (nonprescription) acne products for several weeks and they haven't helped, ask your doctor about prescription-strength medications. A dermatologist can help you.",
#         'Diabetes': "Type 1 diabetes: Treatment aims at maintaining normal blood sugar levels through regular monitoring, insulin therapy, diet and exercise. Type 2 diabetes: Treatments include diet, exercise, medication and insulin therapy. Prediabetes: Progression from prediabetes to type 2 diabetes isn't inevitable. With lifestyle changes, weight loss and medication, it's possible to bring a blood sugar level back to normal.",
#         'Impetigo': 'Impetigo is treated with prescription mupirocin antibiotic ointment or cream applied directly to the sores two to three times a day for five to 10 days. Before applying the medicine, soak the area in warm water or apply a wet cloth compress for a few minutes. Then pat dry and gently remove any scabs so the antibiotic can get into the skin. Place a nonstick bandage over the area to help prevent the sores from spreading.',
#         'Hypertension': 'Changing your lifestyle can help control and manage high blood pressure. Your health care provider may recommend that you make lifestyle changes including: Eating a heart-healthy diet with less salt, Getting regular physical activity, Maintaining a healthy weight or losing weight, No alcohol, No smoking and 7 to 9 hours of sleep daily.',
#         'Peptic ulcer diseae': 'Treatment for peptic ulcers depends on the cause. Usually, treatment will involve killing the H. pylori bacterium if present, eliminating or reducing use of NSAIDs if possible, and helping your ulcer to heal with medication.',
#         'Dimorphic hemorrhoids(piles)': 'You can often relieve the mild pain, swelling and inflammation of hemorrhoids with home treatments. Eat high-fiber foods. Eat more fruits, vegetables and whole grains. Doing so softens the stool and increases its bulk, which will help you avoid the straining that can worsen symptoms from existing hemorrhoids. Add fiber to your diet slowly to avoid problems with gas. You can also take oral pain relievers such as acetaminophen (Tylenol, others), aspirin or ibuprofen (Advil, Motrin IB, others) temporarily to help relieve your discomfort.',
#         'Common Cold': 'Stay hydrated. Water, juice, clear broth or warm lemon water with honey helps loosen congestion and prevents dehydration. Avoid alcohol, coffee and caffeinated sodas, which can make dehydration worse. Rest. Your body needs rest to heal. Combat stuffiness. Over-the-counter saline nasal drops and sprays can help relieve stuffiness and congestion. Over-the-counter (OTC) cold and cough medications. For adults and children age 5 and older, OTC decongestants, antihistamines and pain relievers might offer some symptom relief. ',
#         'Chicken pox': 'In otherwise healthy children, chickenpox typically needs no medical treatment. Your doctor may prescribe an antihistamine to relieve itching. But for the most part, the disease is allowed to run its course.',
#         'Cervical spondylosis': "Treatment for cervical spondylosis depends on its severity. The goal of treatment is to relieve pain, help you maintain your usual activities as much as possible, and prevent permanent injury to the spinal cord and nerves. If nonprescription pain relievers aren't enough, your health care provider might prescribe: Nonsteroidal anti-inflammatory drugs, Corticosteroids, Muscle relaxants, Anti-seizure medications and Antidepressants.",
#         'Hyperthyroidism': 'There are several treatments available for hyperthyroidism. The best approach for you depends on your age and health and the severity of hyperthyroidism. Your personal preference also should be considered as you and your health care provider decide on a treatment plan. Treatment may include: Anti-thyroid medicine, Beta blockers, Radioiodine therapy and Thyroidectomy.',
#         'Urinary tract infection': 'Antibiotics usually are the first treatment for urinary tract infections. Your health and the type of bacteria found in your urine determine which medicine is used and how long you need to take it. Medicines commonly used for simple UTIs include: Trimethoprim and sulfamethoxazole (Bactrim, Bactrim DS), Fosfomycin (Monurol), Nitrofurantoin (Macrodantin, Macrobid, Furadantin) and Cephalexin.',
#         'Varicose veins': 'Treatment for varicose veins may include self-care measures, compression stockings, and surgeries or procedures. Procedures to treat varicose veins are often done as an outpatient procedure, which means you usually go home on the same day.',
#         'AIDS': "Currently, there's no cure for HIV/AIDS. Once you have the infection, your body can't get rid of it. However, there are many medications that can control HIV and prevent complications. These medications are called antiretroviral therapy (ART). Everyone diagnosed with HIV should be started on ART, regardless of their stage of infection or complications.",
#         'Paralysis (brain hemorrhage)': 'Treatment focuses on stabilizing your condition, treating an aneurysm if you have one, and preventing complications. If your bleeding is caused by a ruptured brain aneurysm, the medical professionals might recommend: Surgery or Endovascular embolization.',
#         'Typhoid': 'Antibiotic therapy is the only effective treatment for typhoid fever. Fluoroquinolones - These antibiotics, including ciprofloxacin (Cipro), may be a first choice. They stop bacteria from copying themselves. But some strains of bacteria can live through treatment. These bacteria are called antibiotic resistant. Cephalosporins - This group of antibiotics keeps bacteria from building cell walls. One kind, ceftriaxone, is used if there is antibiotic resistance. Macrolides - This group of antibiotics keeps bacteria from making proteins. One kind called azithromycin (Zithromax) can be used if there is antibiotic resistance.',
#         'Hepatitis B': "If you know you've been exposed to the hepatitis B virus, call your health care provider immediately. An injection of immunoglobulin (an antibody) given within 24 hours of exposure to the virus may help protect you from getting sick with hepatitis B. Because this treatment only provides short-term protection, you also should get the hepatitis B vaccine at the same time if you never received it.",
#         'Fungal infection': 'Treatment usually involves antifungal medications that you put on your skin. You might use an over-the-counter cream such as: Clotrimazole, Miconazole and Terbinafine.',
#         'Hepatitis C': 'Antiviral medications - Hepatitis C infection is treated with antiviral medications intended to clear the virus from your body. The goal of treatment is to have no hepatitis C virus detected in your body at least 12 weeks after you complete treatment. Liver transplantation - If you have developed serious complications from chronic hepatitis C infection, liver transplantation may be an option. During liver transplantation, the surgeon removes your damaged liver and replaces it with a healthy liver. Vaccinations - Although there is no vaccine for hepatitis C, your doctor will likely recommend that you receive vaccines against the hepatitis A and B viruses. These are separate viruses that also can cause liver damage and complicate the course of chronic hepatitis C.',
#         'Migraine': 'Migraine treatment is aimed at stopping symptoms and preventing future attacks.  Medications have been designed to treat migraines. Medications used to combat migraines fall into two broad categories: Pain-relieving medications. Also known as acute or abortive treatment, these types of drugs are taken during migraine attacks and are designed to stop symptoms. Preventive medications. These types of drugs are taken regularly, often daily, to reduce the severity or frequency of migraines.',
#         'Bronchial Asthma': 'Bronchodilators: These medicines relax the muscles around your airways. The relaxed muscles let the airways move air. Anti-inflammatory medicines: These medicines reduce swelling and mucus production in your airways. Biologic therapies for asthma: These are used for severe asthma when symptoms persist despite proper inhaler therapy.',
#         'Alcoholic hepatitis': 'Treatment for alcoholic hepatitis involves quitting drinking and therapies to ease the signs and symptoms of liver damage. Treatment might include: Medications, Counseling, Alcoholics Anonymous or other support groups and Outpatient or residential treatment program.',
#         'Jaundice': 'Jaundice can lead to itching, or pruritis. A 2021 Trusted article Source notes that a person can have warm baths containing oatmeal and take antihistamines for mild pruritis. A healthcare professional may prescribe medications for those experiencing moderate to severe pruritis, such as cholestyramine or colestipol. As jaundice may sometimes indicate damage to the liver, a liver transplant may be necessary in some cases, depending on the severity of the injury.',
#         'Hepatitis E': 'In most cases, hepatitis E goes away on its own in about 4-6 weeks. These steps can help ease your symptoms: Rest, Eat healthy foods, Drink lots of water and Avoid alcohol.',
#         'Dengue': 'There is no treatment for dengue, but you can help ease your symptoms by: resting, drinking plenty of fluids and taking paracetamol to help bring down your temperature and ease any pain.',
#         'Hepatitis D': 'If you have HDV, you may need to see a doctor who works with diseases of the digestive tract, including the liver, such as a gastroenterologist. Doctors called hepatologists to specialize even further and treat only liver disease. There’s no cure yet for HDV. Until doctors come up with better options, the drug prescribed most often is pegylated interferon alfa (peg-IFNa). Peg-IFNa doesn’t work well for everyone. It can also cause many side effects, like lack of energy, weight loss, flu-like symptoms, and mental health issues like depression.',
#         'Heart attack': 'Aspirin. Aspirin reduces blood clotting. It helps keep blood moving through a narrowed artery. If you called 911 or your local emergency number, you may be told to chew aspirin. Emergency medical providers may give you aspirin immediately. Clot busters (thrombolytics or fibrinolytics) - These drugs help break up any blood clots that are blocking blood flow to the heart. The earlier a thrombolytic drug is given after a heart attack, the less the heart is damaged and the greater the chance of survival.',
#         'Pneumonia': 'Treatment for pneumonia involves curing the infection and preventing complications. People who have community-acquired pneumonia usually can be treated at home with medication. Although most symptoms ease in a few days or weeks, the feeling of tiredness can persist for a month or more. Specific treatments depend on the type and severity of your pneumonia, your age and your overall health. The options include: Antibiotics, Cough medicine, Fever reducers/pain relievers.',
#         'Arthritis': 'Arthritis treatment focuses on relieving symptoms and improving joint function. You may need to try several different treatments, or combinations of treatments, before you determine what works best for you. Commonly used arthritis medications include: NSAIDs. Nonsteroidal anti-inflammatory drugs (NSAIDs) can relieve pain and reduce inflammation. Counterirritants - some varieties of creams and ointments contain menthol or capsaicin, the ingredient that makes hot peppers spicy. Steroids - Corticosteroid medications, such as prednisone, reduce inflammation and pain and slow joint damage. ',
#         'Gastroenteritis': "There's often no specific medical treatment for viral gastroenteritis. Antibiotics aren't effective against viruses. Treatment involves self-care measures, such as staying hydrated and taking rest.",
#         'Tuberculosis': 'For active tuberculosis, you must take antibiotics for at least six to nine months. The exact drugs and length of treatment depend on your age, overall health, possible drug resistance and where the infection is in your body. Most common TB drugs are Isoniazid, Rifampin (Rifadin, Rimactane), Ethambutol (Myambutol) and Pyrazinamide.'}

#     if request.method == 'POST':
#         symptoms = request.form.get('input_symptoms')
#         symptoms = symptoms.split(",")

#         # creating input data for the models
#         input_data = [0] * len(symptom_dictionary["symptom_index"])
#         for symptom in symptoms:
#             index = symptom_dictionary["symptom_index"][symptom]
#             input_data[index] = 1

#         # reshape input_data to numpy array
#         input_data = np.array(input_data).reshape(1, -1)

        # generating individual outputs
    #     prediction = symptom_dictionary["predictions_classes"][model.predict(input_data)[0]]
    #     send_explanation = disease_with_explanation[prediction]
    #     send_symptoms = disease_with_symptoms[prediction]
    #     send_treatment = disease_with_treatment[prediction]

    #     print(jsonify({'prediction': prediction, 'explanation': send_explanation, 'symptoms': send_symptoms,
    #                    'treatment': send_treatment}))
    #     if prediction and send_explanation and send_symptoms and send_treatment:
    #         return jsonify({'prediction': prediction, 'explanation': send_explanation, 'symptoms': send_symptoms,
    #                         'treatment': send_treatment})
    #     return jsonify({'error': 'Missing data!'})
   


@app.route('/mental', methods=['GET', 'POST'])
def mental_predict():
    if request.method == 'POST':
        mental_report = request.form.get('mental_report')
        print(mental_report)
        my_string = mental_report
        mental_list = [int(val) if val != '' else 0 for val in my_string.split(",")]

        # Reshape the input data to be two-dimensional
        mental_array = np.array(mental_list).reshape(1, -1)
        print(mental_array)
        mental_prediction = model2.predict(mental_array)
        print(mental_prediction)
        mental_prediction = np.round(mental_prediction)
        print(mental_prediction)
    return jsonify({'mental_report': mental_prediction.tolist()})


if __name__ == "__main__":
    app.run(debug=True)
