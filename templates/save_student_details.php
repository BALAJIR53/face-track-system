<?php
$servername = "localhost";
$username = "root"; // your MySQL username
$password = ""; // your MySQL password
$dbname = "attendanceDB"; // your database name

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Check if the form is submitted
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    // Get form data
    $roll_number = $_POST['roll-number'];
    $department = $_POST['department'];
    $year_of_study = $_POST['year-of-study'];

    // Insert into database (changed table name to login_history)
    $sql = "INSERT INTO login_history (roll_number, department, year_of_study, date, status) 
            VALUES ('$roll_number', '$department', '$year_of_study', CURDATE(), 'Pending')";

    if ($conn->query($sql) === TRUE) {
        echo "Student details saved successfully.";
    } else {
        echo "Error: " . $sql . "<br>" . $conn->error;
    }
}

// Close the connection
$conn->close();
?>
