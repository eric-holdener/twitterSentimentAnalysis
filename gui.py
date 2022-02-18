from PyQt5.QtWidgets import *

# use this file to create the gui to select # of tweets to scrape, keyword, and filename - beginning
# spit out a wordcloud net, as well as visualizations on the sentiment in the gui

def runGUI():
    # every gui app needs one QApplication
    # [] represent command line arguments passed to the application
    app = QApplication([])

    userInput(app)

def userInput(app):


    # define the window to open and the layout of the window
    window = QWidget()
    layout = QVBoxLayout()

    # define the widgets we want in this window
    button = QPushButton('Go')

    layout.addWidget(QLineEdit('Keyword'))
    layout.addWidget(QLineEdit('# of Tweets to Scrape'))
    layout.addWidget(button)

    # set the layout of the window to the layout defined earlier
    window.setLayout(layout)

    # show the window
    window.show()

    button.clicked.connect(on_button_clicked())

    # # run the application until the user closes it
    app.exec_()

def on_button_clicked():
    alert = QMessageBox()
    alert.setText('You clicked the button!')
    alert.exec()

if __name__ == '__main__':
    runGUI()