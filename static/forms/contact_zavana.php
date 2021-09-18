<?php
	$name = $_POST ['name'];
	$visitor_email = $_POST['email'];
	$visitor_subject = $_POST['subject'];
	$message = $_POST ['message'];

	$email_from = 'ZAVANA - STOCK RECOMMENDATION AND PREDICTION';
	$email_subject = "New Form Submission";
	$email_body = "User Name: $name.\n".
					"User Email: $visitor_email.\n".
						"User Subject: $visitor_subject.\n".
							"User Message: $message.\n";

	$to = "zavanacontact@gmail.com";
	$headers = "From: $email_from \r\n";
	$headers .= "Reply-To: $visitor_email \r\n";

	mail($to,$email_subject,$email_body,$headers);
	header("Location: index.html");
?>