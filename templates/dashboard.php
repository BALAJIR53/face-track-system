<?php
header('Content-Type: application/json');

// Database credentials
$servername = "localhost";
$username = "root";  // default XAMPP MySQL user
$password = "";      // default password is empty
$dbname = "userDB";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    http_response_code(500);
    echo json_encode(["error" => "Connection failed: " . $conn->connect_error]);
    exit();
}

// Query attendance data (adjust table and column names accordingly)
$sql = "SELECT userid, username, status, timestamp FROM attendance ORDER BY timestamp DESC";

$result = $conn->query($sql);

$attendanceData = [];

if ($result->num_rows > 0) {
    while($row = $result->fetch_assoc()) {
        $attendanceData[] = $row;
    }
}

echo json_encode($attendanceData);

$conn->close();
?>
