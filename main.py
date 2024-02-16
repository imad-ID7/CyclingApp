import numpy as np
from kivy.app import App
import random
from kivy.properties import ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.config import Config
from kivy.properties import StringProperty
from kivy.properties import ListProperty
from kivy_garden.graph import Graph, LinePlot
from kivy.uix.screenmanager import ScreenManager, Screen
from os.path import exists
import json
from sklearn.linear_model import LinearRegression
from kivy.clock import Clock
from kivy.garden.graph import LinePlot
from kivy.properties import NumericProperty


# Data
Velocity_data = np.random.randint(0, 30, 1000).tolist()         # Replace this array with your actual Velocity data
heart_rate_data = np.random.randint(60, 100, 1000).tolist()      # Replace this array with your actual Heart rate data
real_time_Velocity = Velocity_data[-1]
real_time_heart_rate = heart_rate_data[-1]
zone_heart_rate_values = {
    1: {"min": 55, "max": 65},
    2: {"min": 65, "max": 75},
    3: {"min": 80, "max": 85},
    4: {"min": 85, "max": 88},
    5: {"min": 90, "max": 100},
    }


# User Login
current_username = 'imad'
firstname_loaded = ""

# Variables
target_velocity = 0
max_heart_rate = 0
min_percentage = 0
max_percentage = 0
current_zone = ''
min_target_zone = 0
max_target_zone = 0
target_zone_on_heart_rate = [0,0]
previous_screen = None
training = None




# Load the data file
path = "sign_up.json"
file = None
        



# Screen
Config.set('graphics', 'width', "414")
Config.set('graphics', 'height', "736")
Config.set('graphics', 'resizable', "False")


def predict_heart_rate(V_data, HB_data, New_HR, MHR, target_zone_on_persentage):
        # Reshape the data for sklearn
        X = np.array(V_data).reshape(-1, 1)
        y = np.array(HB_data)

        print("Shape of X:", X.shape)
        print("Shape of y:", y.shape)

        # Fit linear regression model
        model = LinearRegression().fit(X, y)

        # Predict heart rate for the desired velocity
        predicted_HB = model.predict(np.array([[New_HR]]))

        # Calculate the range of heart rates in the target zone
        lower_bound = MHR * target_zone_on_persentage[0] / 100
        upper_bound = MHR * target_zone_on_persentage[1] / 100

        # Adjust the predicted heart rate to stay within the target zone
        if predicted_HB < lower_bound:
            target_velocity = New_HR * (lower_bound / predicted_HB)
        elif predicted_HB > upper_bound:
            target_velocity = New_HR * (upper_bound / predicted_HB)
        else:
            target_velocity = New_HR

        return int(target_velocity[0])


def round_to_nearest_half(value):
    # Round the value to the nearest half-integer
    rounded_value = round(value * 2) / 2
    return rounded_value



#------------------Home Page------------------
class HomePage(BoxLayout, Screen):
    firstname_loaded = StringProperty('imad')
    zoom = NumericProperty(1)
    new_update = StringProperty('')
    current_zone = StringProperty('')
    min_target_zone = StringProperty('')
    max_target_zone = StringProperty('')
    target_velocity = StringProperty('')
    start_stop_training = StringProperty("Start Training")
    training = False  # Flag to track training state
    timer_value = 1  # Default timer value in minutes
    timer = 0  # Current timer value in seconds
    start_stop_training_color = ListProperty([0.329, 0.749, 0.596, .5]) # Green color par defalut
    

    def __init__(self, Velocity_data, heart_rate_data,**kwargs):
        super().__init__(**kwargs)
        self.Velocity_data = Velocity_data
        self.heart_rate_data = heart_rate_data
        
    def on_pre_enter(self, *args):
        global training
        training = self.training
        self.current_zone = str(current_zone)
        self.min_target_zone = str(min_target_zone)
        self.max_target_zone = str(max_target_zone)
        self.target_velocity = str(target_velocity)
        self.firstname_loaded = firstname_loaded
        self.real_time_Velocity = real_time_Velocity
        print(current_zone)
        print(min_target_zone)
        print(max_target_zone)

    def start_training(self):
        if not self.training:  # If training is not ongoing
            self.training = True
            self.start_stop_training_color = [0.957, 0.263, 0.212, .5]  # Red color
            self.start_stop_training = "Stop Training"
            print(self.start_stop_training)
            print(self.start_stop_training)
            print("velocity is: {}".format(real_time_Velocity))
            print("max heart rate is: {}".format(max_heart_rate))
            print("zone is: {}".format(tuple(target_zone_on_heart_rate)))
            self.target_velocity = str(predict_heart_rate(self.Velocity_data, self.heart_rate_data, real_time_Velocity, max_heart_rate, tuple(target_zone_on_heart_rate)))

            self.timer = self.timer_value * 60  # Convert minutes to seconds
            Clock.schedule_interval(self.update_timer, 1)  # Start timer
            
            

        else:  # If training is ongoing
            self.stop_training()

    def stop_training(self):
        self.training = False
        self.start_stop_training_color = [0.329, 0.749, 0.596, .5]
        self.start_stop_training = "Start Training"
        print(self.start_stop_training)
        Clock.unschedule(self.update_timer)  # Stop timer
        
        

    def update_timer(self, dt):
        if self.timer > 0:
            self.timer -= 1  # Decrement timer by 1 second
            self.update_ui_timer()  # Update UI with timer value
        else:
            self.timer_finished_action()  # Timer reached 0, handle action

    def update_ui_timer(self):
        minutes = self.timer // 60
        seconds = self.timer % 60
        self.new_update = f"{minutes:02d}:{seconds:02d}"
        print("Time left: {}".format(self.new_update))

    # Add this method to handle the button press
    def on_training_button_press(self):
        if not self.training:  # If training is not ongoing
            self.start_training()  # Start the training process
        else:  # If training is ongoing
            self.stop_training()  # Stop the training process

    def timer_finished_action(self):
        print("Training session finished!")
        # Stop the training process
        self.stop_training()

        # Reset the timer
        self.timer = self.timer_value * 60  # Convert minutes to seconds

        # Restart the training if it was ongoing before
        if not self.training:
            self.start_training()

    def stop_training_external(self):
        if not self.training:  # If training is ongoing
            self.stop_training()  # Stop the training process



    def update_zoom(self, value):
        if value == "+" and self.zoom < 8:
            self.zoom += 2
        elif value == "-" and 2 < self.zoom:
            self.zoom -= 2

    def go_to_velocity_page_button(self):
        self.manager.current = 'velocity'

    def go_to_heart_rate_page_button(self):
        self.manager.current = 'heart_rate'

    def go_to_zones_page_button(self):
        self.manager.current = 'zones'

    def go_to_profile_page_button(self):
        self.manager.current = 'profile'

    def log_out_button(self):
        self.manager.current = 'login_or_signup'
        login_page = App.get_running_app().root.get_screen('login')
        login_page.clear_data()



#------------------Zones Page------------------
class ZonesPage(BoxLayout, Screen):
    def __init__(self, Velocity_data, heart_rate_data,**kwargs):
        super().__init__(**kwargs)
        self.Velocity_data = Velocity_data
        self.heart_rate_data = heart_rate_data

    def on_switch_zone(self,zone_number,active):
        global target_velocity
        global min_target_zone
        global max_target_zone
        global target_zone_on_heart_rate
        global current_zone
        self.current_zone = f"Zone {zone_number}"
        
        print(f"Switch {self.current_zone} is {'on' if active else 'off'}.")
        


        # Turn off all zones
        if active:
            # Get the current username from the login screen
            current_username = App.get_running_app().root.get_screen('login').username_input.text

            # Load the data file
            if exists(path):
                with open(path, 'r') as file:
                    data = json.load(file)

                # Update the "target zone" value for the current user
                if current_username in data:
                    current_zone = self.current_zone
                    data[current_username]["target zone"] = current_zone
                    # Accessing values for Zone
                    min_percentage = zone_heart_rate_values[zone_number]["min"]
                    max_percentage = zone_heart_rate_values[zone_number]["max"]
                    target_zone_on_persentage = (min_percentage, max_percentage)
                    data[current_username]["zone range"] = list(target_zone_on_persentage)
                    target_velocity = predict_heart_rate(self.Velocity_data, self.heart_rate_data, real_time_Velocity, max_heart_rate, target_zone_on_persentage)

                    min_target_zone = round_to_nearest_half((min_percentage/100)*max_heart_rate)
                    max_target_zone = round_to_nearest_half((max_percentage/100)*max_heart_rate)
                    target_zone_on_heart_rate = (min_target_zone,max_target_zone) 
                    data[current_username]["min HR zone"] = min_target_zone
                    data[current_username]["max HR zone"] = max_target_zone
                    print("the target zones for {}% is: {} BPM".format(target_zone_on_persentage,target_zone_on_heart_rate))
                    print(current_zone)

                # Save the updated data back to the file
                with open(path, 'w') as file:
                    json.dump(data, file, indent=4)


            for i in range(1, 6):
                if i != zone_number:
                    switch_id = f'zone{i}'
                    getattr(self.ids, switch_id).active = False
                    # Restart the training process, if a zone has been changed
                    home_page_instance = HomePage(Velocity_data=Velocity_data, heart_rate_data=heart_rate_data)
                    home_page_instance.stop_training_external()
        else:
            # If none of the switches are active, turn on switch1
            if not any(getattr(self.ids, f'zone{i}').active for i in range(1, 6)):
                self.ids.zone1.active = True

    def go_back_to_home_page_button(self):
        self.manager.current = 'home'



#------------------Velocity Page------------------

class VelocityPage(BoxLayout, Screen):
    zoom = NumericProperty(1)
    target_velocity = NumericProperty(0)
    avg_velocity = NumericProperty(0)
    max_velocity = NumericProperty(0)
    real_time_Velocity = NumericProperty(0)

    def __init__(self, Velocity_data, **kwargs):
        super().__init__(**kwargs)
        
        self.Velocity_data = Velocity_data
        self.historical_data = Velocity_data.copy()  # Copy initial data to historical data
        # Initialize the plot attribute
        self.plot = None
        self.training = training

    def on_pre_enter(self, *args):
        global target_velocity
        # Set initial values
        self.target_velocity = target_velocity
        self.avg_velocity = float(np.mean(self.Velocity_data))  # Convert to float
        self.max_velocity = float(np.max(self.Velocity_data))

        
        
        # Print initial values for testing
        print("the MHR is: {}".format(max_heart_rate))
        print("the avg velo is ", self.avg_velocity, " km")
        print("the avg velo is ", self.max_velocity, " km")
        print("the real time HR is: {}".format(real_time_Velocity))
        print("the target velocity is: {}".format(self.target_velocity))

        # Create a more detailed graph with a static x-label
        self.samples = 1500  # Increase number of samples for more detail
        self.graph = Graph(
            xmin=0, xmax=self.samples,
            ymin=0, ymax=max(self.Velocity_data) + 20,
            background_color=[0, 0, 0, 1],  # Dark background
            border_color=[1, 1, 1, 1],       # White border
            tick_color=[1, 1, 1, 0.7],       # Semi-transparent white ticks
            x_grid=True, y_grid=True,
            draw_border=True,                # Draw a border around the graph
            x_grid_label=True, y_grid_label=True,
            x_ticks_major=32, y_ticks_major=20,  # Decrease major tick intervals for more detail
            label_options={'color': [1, 1, 1, 1], 'bold': True},  # White labels
            font_size=14,                    # Larger font size
            xlabel='Time (seconds)',         # Updated label text
            ylabel='Velocity (km/h)'         # Updated label text
        )
        self.ids.graph.add_widget(self.graph)

        # Create LinePlot
        self.plot_x = np.linspace(0, 1, len(self.Velocity_data))
        self.plot_y = self.Velocity_data
        self.plot = LinePlot(color=[1, 1, 0, 1], line_width=1.5)
        self.plot.points = [(x, self.plot_y[x]) for x in range(len(self.Velocity_data))]
        self.graph.add_plot(self.plot)

        self.toggle_training()

    def toggle_training(self):
        if self.training:
            self.training = False
            Clock.schedule_interval(self.update_data, 1)
        else:
            self.training = True
            Clock.unschedule(self.update_data)
    

    def update_data(self, dt):
        global real_time_Velocity
        # Generate new random data
        new_value = random.randint(0, 30)
        self.real_time_Velocity = new_value
        real_time_Velocity = self.real_time_Velocity
        
        # Append the new data point
        self.historical_data.append(new_value)
        
        # Update LinePlot points
        if self.plot is not None:  # Check if plot is initialized
            self.plot_y = np.array(self.historical_data)
            self.plot.points = [(i, val) for i, val in enumerate(self.plot_y)]

    
    def update_zoom(self, value):
        if value == "+" and self.zoom < 8:
            self.zoom += 2
        elif value == "-" and 2 < self.zoom:
            self.zoom -= 2

    def on_pre_leave(self):
        # Clear or reset graphical elements when leaving the page
        self.ids.graph.clear_widgets()

    def go_back_to_home_page_button(self):
        self.manager.current = 'home'



#------------------Heart Rate Page------------------
class Heart_RatePage(BoxLayout, Screen):
    zoom = NumericProperty(1)
    avg_HR = NumericProperty(0)
    max_heart_rate = NumericProperty(0)
    real_time_heart_rate = NumericProperty(0)

    def __init__(self, heart_rate_data,  **kwargs):
        super().__init__(**kwargs)
        self.heart_rate_data = heart_rate_data
        self.real_time_heart_rate = real_time_heart_rate
        self.max_heart_rate = max_heart_rate
        
    def on_pre_enter(self, *args):
        self.max_heart_rate = max_heart_rate
        self.avg_HR = float(np.mean(self.heart_rate_data))  # Convert to float
        print("the avg HR is ", self.avg_HR, " km")

        self.samples = 1000
        self.zoom = 1
        self.graph = Graph(xmin=0, xmax=self.samples,
                           ymin=0, ymax=max(self.heart_rate_data) + 20,
                           background_color=[0, 0, 0, .5],
                           border_color=[0, 1, 1, 1],
                           tick_color=[0, 1, 1, 0.7],
                           x_grid=True, y_grid=True,
                           draw_border=False,
                           x_grid_label=True, y_grid_label=True,
                           x_ticks_major=64, y_ticks_major=20,
                           label_options={'color': [0, 1, 1, 1], 'bold': True},
                           font_size=12,
                           xlabel='Time',
                           ylabel='Heart Rate')
        self.ids.graph.add_widget(self.graph)

        self.plot_x = np.linspace(0, 1, len(self.heart_rate_data))
        self.plot_y = self.heart_rate_data
        self.plot = LinePlot(color=[1, 1, 0, 1], line_width=1.5)
        self.plot.points = [(x, self.plot_y[x]) for x in range(len(self.heart_rate_data))]
        self.graph.add_plot(self.plot)

    def update_zoom(self, value):
        if value == "+" and self.zoom < 8:
            self.zoom += 2
        elif value == "-" and 2 < self.zoom:
            self.zoom -= 2
    def on_pre_leave(self):
        # Clear or reset graphical elements when leaving the page
        self.ids.graph.clear_widgets()


    def go_back_to_home_page_button(self):
        self.manager.current = 'home'



#------------------Login or Signup Page------------------
class LogIn_SignUpPage(BoxLayout, Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def login_button(self):
        self.manager.current = 'login'

    def signup_button(self):
        self.manager.current = 'signup'



#------------------Save Profile Page------------------
class ProfilePage(BoxLayout, Screen):
    firstname_input = ObjectProperty(None)
    lastname_input = ObjectProperty(None)
    age_input = ObjectProperty(None)
    weight_input = ObjectProperty(None)
    gender_input = ObjectProperty(None)
    new_signup_data = None
    file = None
    file_exists = None
    current_username = None
    
    def save_profile_button(self):

        # Get the current username from the SignUpPage
        current_username = App.get_running_app().root.get_screen('signup').username_input.text
        current_password = App.get_running_app().root.get_screen('signup').password_input.text
        print(current_username)

        # firstname_text = self.firstname_input.text
        # lastname_text = self.lastname_input.text #--------------------------- problem
        # age_number = self.age_input.text
        # weight_float = self.weight_input.text
        # gender_text = self.gender_input.text
        firstname_text = "imad1"
        lastname_text = "imad1"
        age_number = "21"
        weight_float = "39.0"
        gender_text = "man"

        # Update the sign_up dictionary with new profile information
        if firstname_text != '' and lastname_text != '' and age_number != '' and weight_float != '' and gender_text != '':
            self.new_signup_data = {
                current_username:{
                    'username': current_username,
                    'password': current_password,
                    'first name': firstname_text,
                    'last name': lastname_text,
                    'age': int(age_number),
                    'weight': float(weight_float),
                    'gender': gender_text,
                    'target zone': 'zone1',
                    'zone range':[80,85],
                    'min HR zone': (55/100)*(208 - 0.7*int(age_number)),
                    'max HR zone': (65/100)*(208 - 0.7*int(age_number)),
                    'mhr':  208 - 0.7*int(age_number),
                }
            }

        # Save the updated sign_up dictionary back to the JSON file
            print(firstname_text,lastname_text,age_number,weight_float,gender_text)
            self.save_to_file()

        # Clear Inputs
            self.firstname_input.text = ''
            self.lastname_input.text = ''
            self.age_input.text = ''
            self.weight_input.text = ''
            self.gender_input.text = ''
            self.rhr_input.text = ''

            signup_page = App.get_running_app().root.get_screen('signup')
            signup_page.clear_data()

    def save_to_file(self):
        self.load_file()
        self.file = open(path, 'r')
        try:
            data = json.load(self.file)
        except json.decoder.JSONDecodeError:
            self.file = open(path, 'w')
            json.dump(self.new_signup_data, self.file, indent=4)
        else:
            self.file = open(path, mode='w')
            data.update(self.new_signup_data)
            json.dump(data, self.file, indent=4)
        finally:
            self.file.close()

    def load_file(self):
        self.file_exists = exists(path)

    def go_back_to_home_page_button(self):
        self.manager.current = 'home'

    def go_back_to_previous_page(self):
        global previous_screen

        # If the user is in the signup process, go back to the signup page
        if previous_screen == 'signup':
            self.manager.current = 'signup'
        # If the user is already signed in, go back to the home page
        else:
            self.manager.current = 'home'


#------------------LogIn Page------------------
class LogInPage(BoxLayout, Screen):
    username_input = ObjectProperty(None)
    password_input = ObjectProperty(None)
    file_exists = None
    something_wrong = StringProperty('')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)   

    def go_to_home_page_button(self):
        global current_username
        global current_zone
        global min_target_zone
        global max_target_zone
        global max_heart_rate
        global firstname_loaded
        global target_zone_on_heart_rate
        global previous_screen
        previous_screen = self.manager.current
        

        username_text = self.username_input.text
        password_text = self.password_input.text

        if username_text == '' and password_text == '' :
            self.something_wrong = 'please enter your username and your password'
        elif password_text == '' :
            self.something_wrong = 'please enter your password'
        else:
            try:
                self.file = open(path, 'r')
                file = open(path, 'r')
                data = json.load(file)
                username_loaded = data[username_text]["username"]
                password_loaded = data[username_text]["password"]
                if username_loaded == username_text and password_loaded == password_text:
                    current_username = username_loaded
                    current_zone = data[current_username]["target zone"]
                    min_target_zone = data[current_username]["min HR zone"]
                    max_target_zone = data[current_username]["max HR zone"]
                    max_heart_rate = data[current_username]["mhr"]
                    firstname_loaded = data[current_username]["first name"]
                    target_zone_on_heart_rate = data[current_username]["zone range"]
                    self.manager.current = 'home'
                elif username_loaded == username_text and password_loaded != password_text:
                    self.something_wrong = 'Your password is wrong!!'
                    print(self.something_wrong)
                else:
                    self.something_wrong = "username doesn't exist!!"
                    print(self.something_wrong)
            except Exception:
                print("username doesn't exist")
                self.something_wrong = "username doesn't exist"
            finally:
                pass

    def clear_data(self):
        self.username_input.text = ''
        self.password_input.text = ''    


#------------------Signup Page------------------
class SignUpPage(BoxLayout, Screen):
    username_input = ObjectProperty(None)
    password_input = ObjectProperty(None)
    new_signup_data = None
    file = None
    file_exists = None
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def save_userpass_button(self):
        global previous_screen
        username_text = self.username_input.text
        password_text = self.password_input.text

        if username_text != '' and password_text != '':
            self.new_signup_data = {
                username_text:{
                    'username': username_text,
                    'password': password_text,
                    'first name': '',
                    'last name': '',
                    'age': 0,
                    'weight': 0.0,
                    'gender': '',
                    'rhr': 0,
                    'target zone': '',
                }
            }

            print(username_text,password_text)
            self.save_to_file()
            previous_screen = self.manager.current
            self.manager.current = 'profile'
            
    def clear_data(self):
        # Clear input fields
        self.username_input.text = ''
        self.password_input.text = ''

    def save_to_file(self):
        self.load_file()
        self.file = open(path, 'r')
        try:
            data = json.load(self.file)
        except json.decoder.JSONDecodeError:
            self.file = open(path, 'w')
            json.dump(self.new_signup_data, self.file, indent=4)
        else:
            self.file = open(path, mode='w')
            data.update(self.new_signup_data)
            json.dump(data, self.file, indent=4)
        finally:
            self.file.close()

    def create_file(self):
        self.file = open(path, 'x')

    def load_file(self):
        self.file_exists = exists(path)
        if not self.file_exists:
            self.create_file()

    def go_back_to_login_or_singup_page(self):
        self.manager.current = 'login_or_signup'



class CyclistApp(App):
    input_color = 1,1,1,0.1
    btn_color = 1,1,1,0.2
    white = 1,1,1,1
    background_path = r"C:\Users\imad-\Desktop\CyclingApp\backgound_image11.jpg"
    Label_Background_Color = 0, 0, 0, .5

    def build(self):
        Screen_Manager = ScreenManager()

        # Create and add screens to ScreenManager
        Screen_Manager.add_widget(LogIn_SignUpPage(name='login_or_signup'))
        Screen_Manager.add_widget(LogInPage(name='login'))        
        Screen_Manager.add_widget(HomePage(name='home', Velocity_data=Velocity_data, heart_rate_data=heart_rate_data))
        Screen_Manager.add_widget(ZonesPage(name="zones", Velocity_data=Velocity_data, heart_rate_data=heart_rate_data))
        Screen_Manager.add_widget(Heart_RatePage(name='heart_rate', heart_rate_data=heart_rate_data))
        Screen_Manager.add_widget(VelocityPage(name='velocity', Velocity_data=Velocity_data))
        Screen_Manager.add_widget(SignUpPage(name='signup'))
        Screen_Manager.add_widget(ProfilePage(name='profile'))

        return Screen_Manager


CyclistApp().run()