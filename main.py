import kivy
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput


import config
import mlp
from init import init_file

data = []

class Main(GridLayout):

    def __init__(self, **kwargs):
        super(Main, self).__init__(**kwargs)
        self.cols = 2

        #################################
        divLeft = BoxLayout(orientation='vertical')
        #
        navBar = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        nav_train_btn = Button(text='Train', background_color='skyblue')
        nav_train_btn.bind(on_press=self.handle_nav_train)
        navBar.add_widget(nav_train_btn)
        nav_test_btn = Button(text='Test', background_color='skyblue')
        nav_test_btn.bind(on_press=self.handle_nav_test)
        navBar.add_widget(nav_test_btn)
        #
        charGrid = GridLayout()
        charGrid.rows = 5
        charGrid.cols = 5
        for i in range(5):
            for j in range(5):
                btn = Button(background_color=config.btn_inactive_bg)
                btn.bind(on_press = lambda btn, row=i, col=j: self.handle_press(btn, (row,col)))
                charGrid.add_widget(btn)
        #
        divLeft.add_widget(navBar)
        divLeft.add_widget(charGrid)

        ####
        self.divRight = BoxLayout(orientation='vertical')
        self.show_test()
        #################################
        self.add_widget(divLeft)    
        self.add_widget(self.divRight)

    def train(self, instance): 
        #here must run the mlp algoritm to train the network:
        mlp.train()

    def test(self, instance):
        prediction = mlp.test(data)
        self.output.text = prediction

    def btn_is_active(self, btn):
        bg = btn.background_color
        active = config.btn_active_bg
        if bg[0] == active[0] and bg[1] == active[1] and bg[2] == active[2] and bg[3] == active[3]:
            return True
        else:
            return False   

    def handle_press(self, instance, coordinate):
        if self.btn_is_active(instance):
            instance.background_color = config.btn_inactive_bg
            data[coordinate[0]][coordinate[1]] = -1
        else:
            instance.background_color = config.btn_active_bg
            data[coordinate[0]][coordinate[1]] = 1

    def handle_nav_train(self, instance):
        self.show_train()
    def handle_nav_test(self, instance):
        self.show_test()

    def show_train(self):
        self.divRight.clear_widgets()
        train_btn = Button(text='train it!', size_hint=(1, 0.15), background_color='orange', color='white')
        train_btn.bind(on_press = self.train)
        self.divRight.add_widget(BoxLayout(size_hint=(1,0.85)))
        self.divRight.add_widget(train_btn)
    
    def show_test(self):
        self.divRight.clear_widgets()
        self.divRight.add_widget(Label(text='the predicted label character is:', size_hint=(1, 0.1)))
        self.output = Label(text='', font_size=200)
        self.divRight.add_widget(self.output)
        test_btn = Button(text='test it!', size_hint=(1, 0.15), background_color='green', color='white')
        test_btn.bind(on_press = self.test)
        self.divRight.add_widget(test_btn)
        

class XO_Recognizer_mlp(App):

    for i in range(5):
        data.append([])
        for j in range(5):
            data[i].append(-1)

    init_file()

    def build (self):
        return Main()

if(__name__ == "__main__"):
    XO_Recognizer_mlp().run()