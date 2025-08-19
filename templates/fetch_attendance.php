<?php
header('Content-Type: application/json');

// Database connection details
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "userDB";

$conn = new mysqli($servername, $username, $password, $dbname);
if ($conn->connect_error) {
    echo json_encode(['error' => "Connection failed: " . $conn->connect_error]);
    exit();
}

$studentId = isset($_GET['studentId']) ? $_GET['studentId'] : '';
$startDate = isset($_GET['startDate']) ? $_GET['startDate'] : '';
$endDate = isset($_GET['endDate']) ? $_GET['endDate'] : '';

$sql = "SELECT id, name, user_id, date, status, timestamp FROM attendance WHERE 1";

if ($studentId !== '') {
    $studentId = $conn->real_escape_string($studentId);
    $sql .= " AND user_id = '$studentId'";
}
if ($startDate !== '') {
    $startDate = $conn->real_escape_string($startDate);
    $sql .= " AND date >= '$startDate'";
}
if ($endDate !== '') {
    $endDate = $conn->real_escape_string($endDate);
    $sql .= " AND date <= '$endDate'";
}

$sql .= " ORDER BY date DESC";

$result = $conn->query($sql);
if (!$result) {
    echo json_encode(['error' => $conn->error]);
    $conn->close();
    exit();
}

$attendance = [];
while ($row = $result->fetch_assoc()) {
    $attendance[] = $row;
}

echo json_encode($attendance);
$conn->close();
?>
