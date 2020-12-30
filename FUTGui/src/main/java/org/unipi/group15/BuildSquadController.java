package org.unipi.group15;

import bean.Player;
import bean.Squad;
import javafx.beans.binding.Bindings;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;
import mongo.ProvaQuery;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

public class BuildSquadController {
    private static int squadIdex = -1;
    private static Squad squad;

    @FXML private TextField squadNameTextField;

    @FXML private ChoiceBox<String> moduleChoiceBox;

    @FXML private ChoiceBox<String> positionChoiceBox;

    @FXML private TableView<Player> findPlayersTableView;

    @FXML private TableView<Player> chosenPlayersTableView;

    @FXML private TextField findPlayerTextField;

    @FXML private Button addPlayerButton;

    @FXML
    private void initialize(){

        moduleChoiceBox.getItems().removeAll(moduleChoiceBox.getItems());
        moduleChoiceBox.getItems().addAll(FXCollections.observableArrayList("352",
                                            "4231", "4312", "433", "442"));

        chosenPlayersTableView.getColumns().get(0).setCellValueFactory(new PropertyValueFactory<>("playerName"));
        chosenPlayersTableView.getColumns().get(1).setCellValueFactory(new PropertyValueFactory<>("pace"));
        chosenPlayersTableView.getColumns().get(2).setCellValueFactory(new PropertyValueFactory<>("shooting"));
        chosenPlayersTableView.getColumns().get(3).setCellValueFactory(new PropertyValueFactory<>("passing"));
        chosenPlayersTableView.getColumns().get(4).setCellValueFactory(new PropertyValueFactory<>("defending"));
        chosenPlayersTableView.getColumns().get(5).setCellValueFactory(new PropertyValueFactory<>("physicality"));

        findPlayersTableView.getColumns().get(0).setCellValueFactory(new PropertyValueFactory<>("playerExtendedName"));
        findPlayersTableView.getColumns().get(1).setCellValueFactory(new PropertyValueFactory<>("overall"));
        findPlayersTableView.getColumns().get(2).setCellValueFactory(new PropertyValueFactory<>("quality"));
        findPlayersTableView.getColumns().get(3).setCellValueFactory(new PropertyValueFactory<>("revision"));
        findPlayersTableView.getColumns().get(4).setCellValueFactory(new PropertyValueFactory<>("club"));

        if(squadIdex != -1) {
            squad = App.getSession().getSquads().get(squadIdex);
            squadNameTextField.setText(squad.getName());
            moduleChoiceBox.getSelectionModel().select(squad.getModule());
            displayModulePositions(squad.getModule());
            ObservableList<Player> players = FXCollections.observableArrayList(squad.getPlayers().values());
            chosenPlayersTableView.setItems(players);
        }
        else {
            squad = new Squad();
        }

        moduleChoiceBox.getSelectionModel().selectedIndexProperty().addListener(new ChangeListener<Number>() {
            @Override
            public void changed(ObservableValue<? extends Number> observableValue, Number oldValue, Number newValue) {
                squad.getPlayers().clear();
                displayModulePositions(moduleChoiceBox.getItems().get((Integer) newValue));
            }
        });

    }

    public void setSquadIndex(int index){ squadIdex = index; }

    private void displayModulePositions(String module){

        squad.setModule(module);
        switch (module){
            case "352":
                ArrayList<String> m352 = new ArrayList(Arrays.asList("GK", "CB0", "CB1", "CB2", "CDM0", "CDM1", "CAM1", "LM", "RM", "ST0", "ST1"));
                displayPosition(m352);
                break;
            case "4231":
                ArrayList<String> m4231 = new ArrayList(Arrays.asList("GK","CB0","CB1","LB","RB","CAM1","CDM0","CDM1","CAM0","CAM2","ST"));
                displayPosition(m4231);
                break;
            case "4312":
                ArrayList<String> m4312 = new ArrayList(Arrays.asList("GK","CB0","CB1","LB","RB","CM0","CM1","CM2","CAM1","ST0","ST1"));
                displayPosition(m4312);
                break;
            case "433":
                ArrayList<String> m433 = new ArrayList(Arrays.asList("GK","CB0","CB1","LB","RB","CM0","CM1","CM2","ST0","LW","RW"));
                displayPosition(m433);
                break;
            case "442":
                ArrayList<String> m442 = new ArrayList(Arrays.asList("GK","CB0","CB1","LB","RB","CM0","CM1","LM","RM","ST0","ST1"));
                displayPosition(m442);
                break;
            default: break;
        }
    }

    @FXML
    private void switchToProfile() throws IOException { App.setRoot("userPage"); }

    private void displayPosition(ArrayList<String> elem){
        choosePlayerBox(elem);
        chosenPlayersTableView.getItems().clear();

    }

    private void choosePlayerBox(ArrayList<String> elem){
        positionChoiceBox.getItems().removeAll(positionChoiceBox.getItems());
        positionChoiceBox.getItems().addAll(FXCollections.observableArrayList(elem));
    }

    @FXML
    private void setSquadNane(){
        squad.setName(squadNameTextField.getText());
    }

    @FXML
    private void selectPlayer(){
        if(findPlayerTextField.getText().equals(""))
            return;

        ProvaQuery pq = new ProvaQuery();
        findPlayersTableView.getItems().clear();
        ObservableList<Player> players = FXCollections.observableArrayList(pq.findPlayers(findPlayerTextField.getText()));
        findPlayersTableView.setItems(players);
        findPlayersTableView.setFixedCellSize(25);
        findPlayersTableView.prefHeightProperty().bind(Bindings.size(findPlayersTableView.getItems()).multiply(findPlayersTableView.getFixedCellSize()).add(30));
    }

    @FXML
    private void addPlayer(){
        Player player = (Player) findPlayersTableView.getSelectionModel().getSelectedItem();
        String pos = (String) positionChoiceBox.getSelectionModel().getSelectedItem();
        squad.getPlayers().put(pos, player);
        ObservableList<Player> players = FXCollections.observableArrayList(squad.getPlayers().values());
        chosenPlayersTableView.setItems(players);
    }

    @FXML
    private void saveSquad(){

    }
}